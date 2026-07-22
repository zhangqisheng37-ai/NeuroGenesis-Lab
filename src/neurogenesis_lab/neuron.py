from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, ClassVar
from uuid import uuid4

from neurogenesis_lab._components import ComponentMixin
from neurogenesis_lab.config import (
    HERITABLE_ATTRS,
    SPAWN_POSITION_JITTER,
    NOVELTY_DECAY_STEP,
    NOVELTY_GROWTH_STEP,
    STATE_NATURAL_DECAY,
    REWARD_EMOTION_SCALE,
    RECENT_TOKEN_WINDOW,
    NEURON_MOVE_JITTER,
)
from neurogenesis_lab.encoding import Event


def clamp(value: float, minimum: float, maximum: float) -> float:
    """将数值限制在指定范围内。

    Args:
        value: 待限制的数值。
        minimum: 下界。
        maximum: 上界。

    Returns:
        限制后的数值，保证 minimum <= 返回值 <= maximum。

    Example:
        >>> clamp(1.5, 0.0, 1.0)
        1.0
    """
    return max(minimum, min(maximum, value))


def safe_float(value: float, fallback: float = 0.0) -> float:
    """防止 NaN/Inf 污染，返回安全的有限浮点数。

    Args:
        value: 待检查的浮点数。
        fallback: 当 value 为 NaN 或 Inf 时的回退值，默认 0.0。

    Returns:
        若 value 有限则原样返回，否则返回 fallback。

    Example:
        >>> safe_float(float('nan'), -65.0)
        -65.0
    """
    if math.isfinite(value):
        return value
    return fallback


def calculate_reward(novelty: float, energy_loss: float) -> float:
    """根据新颖度和能量损耗计算奖励值。

    Args:
        novelty: 新颖度得分，范围 [0, 1]。
        energy_loss: 本 tick 的能量损耗（非负）。

    Returns:
        奖励值 = novelty - energy_loss，经 safe_float 保护。
    """
    return safe_float(novelty - energy_loss)


@dataclass
class NeuronDynamics:
    """神经元动力学参数 — 从 Neuron 中提取的纯数据组件。

    Attributes:
        v: 膜电位，单位 mV，默认 -65.0。
        u: 恢复变量（Izhikevich 模型使用），默认 0.0。
        w: 适应电流（AdEx 模型使用），默认 0.0。
        v_rest: 静息电位，单位 mV，默认 -65.0。
        v_reset: 重置电位，单位 mV，默认 -65.0。
        v_th: 放电阈值，单位 mV，默认 -55.0。
        tau_m: 膜时间常数，单位 ms，必须 > 0，默认 20.0。
        r_m: 膜电阻，无量纲缩放因子，默认 1.0。
        spike_cost: 每次放电的能量消耗，>= 0，默认 0.05。
        input_gain: 接收事件时的能量增益，>= 0，默认 0.05。
        move_cost: 每次移动的能量消耗，>= 0，默认 0.01。
    """

    v: float = -65.0
    u: float = 0.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_th: float = -55.0
    tau_m: float = 20.0
    r_m: float = 1.0
    spike_cost: float = 0.05
    input_gain: float = 0.05
    move_cost: float = 0.01


@dataclass
class NeuronEvolution:
    """神经元演化统计 — 从 Neuron 中提取的计数器组件。

    Attributes:
        spike_count: 累计放电次数。
        propagation_count: 累计信号传播次数。
        prediction_hits: 累计预测命中次数（学习引擎使用）。
    """

    spike_count: int = 0
    propagation_count: int = 0
    prediction_hits: int = 0


@dataclass
class NeuronMemory:
    """神经元记忆组件 — 从 Neuron 中提取的事件缓冲区。

    Attributes:
        mem_size: 记忆缓冲区容量（事件条数），默认 64。
        mem: 事件环形缓冲区（deque），最大长度为 mem_size。
        transitions: token 二元组转移计数器，键为 (prev_token, cur_token)。
        _observe_cache: observe() 方法的缓存结果。
        _observe_dirty: 缓存是否需要刷新。
    """

    mem_size: int = 64
    mem: deque = field(default_factory=lambda: deque(maxlen=64))
    transitions: Counter = field(default_factory=Counter)
    _observe_cache: list[dict[str, int]] = field(default_factory=list)
    _observe_dirty: bool = True


_NEURON_DYNAMICS_FIELDS = frozenset(f.name for f in NeuronDynamics.__dataclass_fields__.values())
_NEURON_EVOLUTION_FIELDS = frozenset(f.name for f in NeuronEvolution.__dataclass_fields__.values())
_NEURON_MEMORY_FIELDS = frozenset(f.name for f in NeuronMemory.__dataclass_fields__.values())


@dataclass
class Neuron(ComponentMixin):
    """基础神经元 — LIF 模型，支持演化、记忆和空间移动。

    内部状态通过 NeuronDynamics、NeuronEvolution、NeuronMemory 三个组件管理，
    对外通过 ``ComponentMixin`` 透明代理访问。

    Attributes:
        id: 唯一标识符，默认 UUID hex。
        pos: 二维空间坐标 (x, y)。
        energy: 当前能量值，<= 0 时死亡。
        alive: 是否存活。
        links: 出边字典，键为目标神经元 ID，值含 weight 和 delay。
        states: 内部状态字典，包含 energy/entropy/emotion/novelty。
        last_energy: 上一 tick 开始时的能量快照。
        last_event: 最近接收的事件。
        last_reward: 最近一次奖励值。

    Example:
        >>> neuron = Neuron(pos=(1.0, 2.0), energy=1.5)
        >>> spiked = neuron.update(dt=1.0, input_current=5.0)
    """

    _component_specs: ClassVar = [
        ("_dynamics", NeuronDynamics, _NEURON_DYNAMICS_FIELDS),
        ("_evolution", NeuronEvolution, _NEURON_EVOLUTION_FIELDS),
        ("_memory", NeuronMemory, _NEURON_MEMORY_FIELDS),
    ]

    id: str = field(default_factory=lambda: uuid4().hex)
    pos: tuple[float, float] = (0.0, 0.0)
    energy: float = 1.0
    alive: bool = True
    links: dict[str, dict[str, float]] = field(default_factory=dict)
    # 以下字段在 __post_init__ 中转移到组件，保留声明以兼容 dataclass 构造签名
    mem_size: int = 64
    mem: deque[Event] = field(default_factory=lambda: deque(maxlen=64))
    transitions: Counter[tuple[int, int]] = field(default_factory=Counter)
    states: dict[str, float] = field(
        default_factory=lambda: {
            "energy": 1.0,
            "entropy": 0.0,
            "emotion": 0.0,
            "novelty": 0.0,
        }
    )
    last_energy: float = 1.0
    last_event: Event | None = None
    last_reward: float = 0.0
    v: float = -65.0
    u: float = 0.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_th: float = -55.0
    tau_m: float = 20.0
    r_m: float = 1.0
    spike_cost: float = 0.05
    input_gain: float = 0.05
    move_cost: float = 0.01
    spike_count: int = 0
    propagation_count: int = 0
    prediction_hits: int = 0
    _observe_cache: list[dict[str, int]] = field(default_factory=list, init=False, repr=False)
    _observe_dirty: bool = field(default=True, init=False, repr=False)
    _dynamics: NeuronDynamics = field(init=False, repr=False)
    _evolution: NeuronEvolution = field(init=False, repr=False)
    _memory: NeuronMemory = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        # 重新初始化 deque 确保 maxlen 生效
        memory = self._memory
        memory.mem = deque(memory.mem, maxlen=memory.mem_size)
        self.states["energy"] = self.energy
        self.last_energy = self.energy
        self._observe_dirty = True

    def start_tick(self) -> None:
        """记录 tick 开始时的能量快照，用于后续奖励计算。"""
        self.last_energy = self.energy

    def receive(self, event: Event) -> list[Event]:
        """接收一个输入事件，更新记忆、转移计数和新颖度。

        Args:
            event: 输入事件。

        Returns:
            空列表（保留接口兼容性）。
        """
        if not self.alive:
            return []
        recent_tokens = [item.token_id for item in list(self.mem)[-RECENT_TOKEN_WINDOW:]]
        self.mem.append(event)
        self.last_event = event
        self.energy = max(0.0, self.energy + self.input_gain)
        if len(self.mem) >= 2:
            prev = self.mem[-2]
            self.transitions[(prev.token_id, event.token_id)] += 1
            self._observe_dirty = True
        if event.token_id in recent_tokens:
            self.states["novelty"] = max(0.0, self.states["novelty"] - NOVELTY_DECAY_STEP)
        else:
            self.states["novelty"] = min(1.0, self.states["novelty"] + NOVELTY_GROWTH_STEP)
        return []

    def update(self, dt: float, input_current: float) -> bool:
        """执行一步 LIF 动力学更新。

        Args:
            dt: 时间步长，单位 ms。
            input_current: 外部输入电流（无量纲缩放值）。

        Returns:
            是否产生放电（膜电位达到阈值）。
        """
        input_current = safe_float(input_current)
        dv = (-(self.v - self.v_rest) + self.r_m * input_current) / self.tau_m
        self.v = safe_float(self.v + dv * dt, self.v_reset)
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.on_spike()
            return True
        return False

    def on_spike(self) -> None:
        """放电回调 — 扣除能量并递增放电计数。"""
        self.energy = max(0.0, self.energy - self.spike_cost)
        self.spike_count += 1

    def emit(self, event: Event, current_tick: int | None = None) -> list[Event]:
        """向所有出边邻居传播事件。

        Args:
            event: 触发传播的源事件。
            current_tick: 当前 tick，若为 None 则使用 event.t。

        Returns:
            生成的传播事件列表，每个邻居一个。
        """
        emitted: list[Event] = []
        if not self.links:
            return emitted
        base_tick = event.t if current_tick is None else current_tick
        for neighbor_id, link in self.links.items():
            weight = float(link.get("weight", 1.0))
            delay = int(link.get("delay", 0))
            if weight <= 0:
                continue
            if delay < 0:
                delay = 0
            self.propagation_count += 1
            emitted.append(
                Event(
                    token_id=event.token_id,
                    t=base_tick + delay,
                    src_id=neighbor_id,
                    meta={"weight": weight, "from": self.id},
                )
            )
        return emitted

    def move(self) -> None:
        """随机移动神经元位置，消耗能量。能量不足时仅扣费不移动。"""
        if not self.alive:
            return
        # 只有能量充足时才移动
        if self.energy > self.move_cost * 2:
            dx = random.uniform(-NEURON_MOVE_JITTER, NEURON_MOVE_JITTER)
            dy = random.uniform(-NEURON_MOVE_JITTER, NEURON_MOVE_JITTER)
            self.pos = (self.pos[0] + dx, self.pos[1] + dy)
            self.energy = max(0.0, self.energy - self.move_cost)
        else:
            self.energy = max(0.0, self.energy - self.move_cost)
        if self.energy <= 0.0:
            self.alive = False

    def apply_reward(self, reward: float) -> None:
        """应用奖励信号，更新情绪状态。

        Args:
            reward: 奖励值，正值提升情绪，负值降低情绪。
        """
        self.last_reward = reward
        self.states["emotion"] = clamp(
            self.states["emotion"] + reward * REWARD_EMOTION_SCALE, 0.0, 1.0
        )

    def finalize_tick(self, had_input: bool) -> None:
        """结束当前 tick，更新能量/熵/情绪/新颖度状态。

        Args:
            had_input: 本 tick 是否接收到输入事件。无输入时状态自然衰减。
        """
        self.states["energy"] = self.energy
        if not had_input:
            self.states["novelty"] *= STATE_NATURAL_DECAY
            self.states["emotion"] *= STATE_NATURAL_DECAY
        tokens = [item.token_id for item in self.mem]
        if not tokens:
            self.states["entropy"] = 0.0
            return
        counts = Counter(tokens)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            ratio = count / total
            entropy -= ratio * math.log(ratio)
        max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
        self.states["entropy"] = entropy / max_entropy if max_entropy > 0 else 0.0

    def observe(self) -> dict[str, Any]:
        """获取神经元可观测状态（带缓存）。

        Returns:
            包含 transition_counts 和 states 的字典。
        """
        if self._observe_dirty:
            top = self.transitions.most_common()
            self._observe_cache = [
                {"from": pair[0], "to": pair[1], "count": count} for pair, count in top
            ]
            self._observe_dirty = False
        return {
            "transition_counts": list(self._observe_cache),
            "states": dict(self.states),
        }

    def _mutate_value(self, value: float, scale: float) -> float:
        if scale <= 0:
            return value
        return value * (1.0 + random.uniform(-scale, scale))

    def fitness(self, weights: dict[str, float] | None = None) -> float:
        """计算适应度得分，用于演化选择。

        Args:
            weights: 各维度权重字典，键为 propagation/energy_efficiency/prediction_hit。
                若为 None 则使用默认权重。

        Returns:
            加权适应度得分（非负浮点数）。

        Example:
            >>> neuron.fitness({"propagation": 1.0, "energy_efficiency": 0.5})
        """
        w = weights or {
            "propagation": 1.0,
            "energy_efficiency": 0.5,
            "prediction_hit": 0.8,
        }
        score = (
            w.get("propagation", 1.0) * self.propagation_count
            + w.get("energy_efficiency", 0.5) * self.energy
            + w.get("prediction_hit", 0.8) * self.prediction_hits
        )
        return safe_float(score)

    def spawn(self, mutation_scale: float, energy_share: float) -> "Neuron":
        """繁殖子代神经元，继承并变异父代参数。

        Args:
            mutation_scale: 变异幅度，范围 [0, 1]，0 表示不变异。
            energy_share: 能量分配比例，范围 (0, 1]，子代获得 energy * share。

        Returns:
            新创建的子代 Neuron 实例。
        """
        child = type(self)()
        child.pos = (
            self.pos[0] + random.uniform(-SPAWN_POSITION_JITTER, SPAWN_POSITION_JITTER),
            self.pos[1] + random.uniform(-SPAWN_POSITION_JITTER, SPAWN_POSITION_JITTER),
        )
        child.energy = max(0.0, self.energy * energy_share)
        child.states = {
            "energy": child.energy,
            "entropy": 0.0,
            "emotion": 0.0,
            "novelty": 0.0,
        }
        for attr in HERITABLE_ATTRS:
            if hasattr(self, attr) and hasattr(child, attr):
                value = getattr(self, attr)
                if isinstance(value, (int, float)):
                    mutated = self._mutate_value(float(value), mutation_scale)
                    setattr(child, attr, safe_float(mutated, float(value)))
        child.v = self.v
        child.u = self.u
        child.w = self.w
        return child


# 删除 dataclass 在类上生成的默认值属性，避免遮蔽 __getattr__ 路由
ComponentMixin.cleanup_dataclass_defaults(Neuron)


@dataclass
class LIFNeuron(Neuron):
    """Leaky Integrate-and-Fire 神经元 — 标准 LIF 模型。

    与基类 Neuron.update() 行为一致，作为显式子类便于类型区分。
    """

    def update(self, dt: float, input_current: float) -> bool:
        """执行一步 LIF 动力学更新。

        Args:
            dt: 时间步长，单位 ms。
            input_current: 外部输入电流。

        Returns:
            是否产生放电。
        """
        input_current = safe_float(input_current)
        dv = (-(self.v - self.v_rest) + self.r_m * input_current) / self.tau_m
        self.v = safe_float(self.v + dv * dt, self.v_reset)
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.on_spike()
            return True
        return False


@dataclass
class IzhikevichNeuron(Neuron):
    """Izhikevich 神经元模型 — 支持多种放电模式。

    通过 (a, b, c, d) 四参数组合可模拟 regular spiking、
    fast spiking、bursting 等多种生物放电模式。

    Attributes:
        a: 恢复变量时间尺度，默认 0.02。
        b: 恢复变量灵敏度，默认 0.2。
        c: 放电后膜电位重置值，单位 mV，默认 -65.0。
        d: 放电后恢复变量增量，默认 8.0。
        v_peak: 放电峰值阈值，单位 mV，默认 30.0。
    """

    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    v_peak: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.u == 0.0:
            self.u = self.b * self.v

    def update(self, dt: float, input_current: float) -> bool:
        """执行一步 Izhikevich 动力学更新。

        Args:
            dt: 时间步长，单位 ms。
            input_current: 外部输入电流。

        Returns:
            是否产生放电（v >= v_peak）。
        """
        input_current = safe_float(input_current)
        dv = 0.04 * self.v * self.v + 5 * self.v + 140 - self.u + input_current
        du = self.a * (self.b * self.v - self.u)
        self.v = safe_float(self.v + dv * dt, self.c)
        self.u = safe_float(self.u + du * dt, self.b * self.c)
        if self.v >= self.v_peak:
            self.v = self.c
            self.u += self.d
            self.on_spike()
            return True
        return False


@dataclass
class AdExNeuron(Neuron):
    """Adaptive Exponential Integrate-and-Fire 神经元模型。

    在 LIF 基础上增加指数项和适应电流 w，
    可模拟 bursting 和 adaptation 等复杂放电行为。

    Attributes:
        delta_t: 指数项锐度参数，单位 mV，默认 2.0。
        a: 亚阈值适应耦合系数，默认 1.0。
        b: 放电后适应电流增量，默认 0.5。
        tau_w: 适应电流时间常数，单位 ms，默认 80.0。
        v_peak: 放电截止阈值，单位 mV，默认 -40.0。
    """

    delta_t: float = 2.0
    a: float = 1.0
    b: float = 0.5
    tau_w: float = 80.0
    v_peak: float = -40.0

    def update(self, dt: float, input_current: float) -> bool:
        """执行一步 AdEx 动力学更新。

        Args:
            dt: 时间步长，单位 ms。
            input_current: 外部输入电流。

        Returns:
            是否产生放电（v >= v_peak）。
        """
        input_current = safe_float(input_current)
        exp_arg = (self.v - self.v_th) / self.delta_t
        # 限制指数参数防止溢出
        exp_arg = min(exp_arg, 20.0)
        exp_term = self.delta_t * math.exp(exp_arg)
        dv = (-(self.v - self.v_rest) + exp_term - self.w + input_current) / self.tau_m
        dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w
        self.v = safe_float(self.v + dv * dt, self.v_reset)
        self.w = safe_float(self.w + dw * dt, 0.0)
        if self.v >= self.v_peak:
            self.v = self.v_reset
            self.w += self.b
            self.on_spike()
            return True
        return False
