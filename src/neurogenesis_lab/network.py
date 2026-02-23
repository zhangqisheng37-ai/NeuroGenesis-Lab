from __future__ import annotations

import heapq
import logging
import math
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from neurogenesis_lab.config import (
    DENSITY_HIGH_THRESHOLD,
    DENSITY_LOW_THRESHOLD,
    DECAY_HIGH_MULTIPLIER,
    DECAY_LOW_MULTIPLIER,
    POPULATION_HIGH_THRESHOLD,
    POPULATION_LOW_THRESHOLD,
    REPRO_SHARE_HIGH_MULTIPLIER,
    REPRO_SHARE_LOW_MULTIPLIER,
    REPRO_SHARE_MAX,
    REPRO_SHARE_MIN,
    RESCUED_NEURON_ENERGY,
    SCHEMA_VERSION,
)
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import Neuron, calculate_reward, safe_float
from neurogenesis_lab.protocols import TickSnapshot, EngineSummary

_logger = logging.getLogger(__name__)


@dataclass
class EventQueue:
    """基于最小堆的事件优先队列，按时间戳排序。

    Attributes:
        _heap: 内部堆存储，元素为 (时间戳, 序号, 事件) 三元组。
        _seq: 单调递增序号，保证同时间戳事件的 FIFO 顺序。
    """

    _heap: list[tuple[int, int, Event]] = field(default_factory=list)
    _seq: int = 0

    def push(self, event: Event) -> None:
        """将事件压入队列。

        Args:
            event: 待入队的事件，按 event.t 排序。
        """
        heapq.heappush(self._heap, (event.t, self._seq, event))
        self._seq += 1

    def push_many(self, events: Iterable[Event]) -> None:
        """批量压入事件。

        Args:
            events: 事件可迭代对象。
        """
        for event in events:
            self.push(event)

    def pop_ready(self, tick: int) -> list[Event]:
        """弹出所有时间戳 <= tick 的就绪事件。

        Args:
            tick: 当前时间步。

        Returns:
            按时间戳升序排列的就绪事件列表。
        """
        ready: list[Event] = []
        while self._heap and self._heap[0][0] <= tick:
            _, _, event = heapq.heappop(self._heap)
            ready.append(event)
        return ready

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class Corpse:
    """死亡神经元遗留的尸体，可被附近存活神经元拾取能量。

    Attributes:
        pos: 尸体空间坐标 (x, y)。
        energy: 剩余可回收能量。
        params: 死亡神经元的动力学参数快照。
        ttl: 剩余存活 tick 数，归零后移除。
        decay_rate: 每 tick 能量自然衰减量，默认 0.02。
    """

    pos: tuple[float, float]
    energy: float
    params: dict[str, float]
    ttl: int
    decay_rate: float = 0.02


@dataclass
class CorpsePool:
    """尸体对象池：复用 Corpse 实例，减少频繁分配。"""

    free: list[Corpse] = field(default_factory=list)
    reused_count: int = 0
    created_count: int = 0

    def acquire(
        self,
        *,
        pos: tuple[float, float],
        energy: float,
        params: dict[str, float],
        ttl: int,
        decay_rate: float = 0.02,
    ) -> Corpse:
        if self.free:
            corpse = self.free.pop()
            corpse.pos = pos
            corpse.energy = energy
            corpse.params = params
            corpse.ttl = ttl
            corpse.decay_rate = decay_rate
            self.reused_count += 1
            return corpse
        self.created_count += 1
        return Corpse(
            pos=pos,
            energy=energy,
            params=params,
            ttl=ttl,
            decay_rate=decay_rate,
        )

    def release(self, corpse: Corpse, max_pool_size: int) -> None:
        if len(self.free) >= max_pool_size:
            return
        corpse.params = {}
        corpse.energy = 0.0
        corpse.ttl = 0
        self.free.append(corpse)


@dataclass
class NetworkConfig:
    """网络配置参数 — 从 Network 中提取的纯配置字段。

    Attributes:
        reproduction_threshold: 繁殖能量阈值，>= 此值才可繁殖，默认 2.5。
        reproduction_share: 繁殖时母体分配给子代的能量比例，范围 (0, 1]，默认 0.4。
        mutation_scale: 变异幅度缩放因子，范围 [0, 1]，默认 0.05。
        energy_decay: 每 tick 基础能量衰减量，>= 0，默认 0.003。
        max_neurons: 网络最大神经元数量，默认 50。
        max_links: 网络最大连接数量，默认 200。
        corpse_ttl: 尸体存活 tick 数，默认 50。
        scavenger_radius: 拾荒搜索半径，默认 2.5。
        input_current_scale: 输入电流缩放因子，默认 1.5。
        background_current: 背景电流，默认 0.0。
        spontaneous_spike_rate: 自发放电概率，范围 [0, 1]，默认 0.0。
        move_cost: 每次移动能量消耗，默认 0.01。
        energy_cap: 能量上限，防止无限积累，默认 5.0。
        adaptive_decay: 是否启用自适应衰减，默认 True。
        adaptive_reproduction: 是否启用自适应繁殖成本，默认 True。
        endangered_threshold: 濒危告警阈值（存活数），默认 10。
        fitness_weights: 适应度各维度权重字典。
        congestion_penalty: 拥塞惩罚概率，范围 [0, 1]，默认 0.1。
        min_neurons: 最小存活神经元数（保护下限），默认 2。
        evolution_mutation_amplitude: 演化变异幅度覆盖值，None 时使用 mutation_scale。
        energy_budget_per_tick: 每 tick 全局能量预算，0 表示不限，默认 0.0。
        schema_version: 数据格式版本号。
        corpse_pool_max_size: 尸体对象池最大容量，默认 2048。
    """

    reproduction_threshold: float = 2.5
    reproduction_share: float = 0.4
    mutation_scale: float = 0.05
    energy_decay: float = 0.003
    max_neurons: int = 50
    max_links: int = 200
    corpse_ttl: int = 50
    scavenger_radius: float = 2.5
    input_current_scale: float = 1.5
    background_current: float = 0.0
    spontaneous_spike_rate: float = 0.0
    move_cost: float = 0.01
    energy_cap: float = 5.0
    adaptive_decay: bool = True
    adaptive_reproduction: bool = True
    endangered_threshold: int = 10
    fitness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "propagation": 1.0,
            "energy_efficiency": 0.5,
            "prediction_hit": 0.8,
        }
    )
    congestion_penalty: float = 0.1
    min_neurons: int = 2
    evolution_mutation_amplitude: float | None = None
    energy_budget_per_tick: float = 0.0
    schema_version: str = SCHEMA_VERSION
    corpse_pool_max_size: int = 2048


@dataclass
class NetworkState:
    """网络运行时统计状态 — 从 Network 中提取的可变状态字段。

    Attributes:
        birth_count: 累计出生数。
        death_count: 累计死亡数。
        total_spikes: 累计放电总数。
        total_propagations: 累计信号传播总数。
        current_tick: 当前 tick。
        _pe_history: 预测误差历史记录。
        _endangered_mode: 是否处于濒危保护模式。
    """

    birth_count: int = 0
    death_count: int = 0
    total_spikes: int = 0
    total_propagations: int = 0
    current_tick: int = 0
    _pe_history: list[float] = field(default_factory=list)
    _endangered_mode: bool = False


# NetworkConfig 和 NetworkState 的字段名集合，用于 __getattr__/__setattr__ 路由
_NETWORK_CONFIG_FIELDS = frozenset(
    f.name for f in NetworkConfig.__dataclass_fields__.values()
)
_NETWORK_STATE_FIELDS = frozenset(
    f.name for f in NetworkState.__dataclass_fields__.values()
)


@dataclass
class Network:
    """神经网络仿真引擎 — 管理神经元集合、事件队列和演化逻辑。

    内部配置和状态分别委托给 NetworkConfig 和 NetworkState 组件，
    通过 ``__getattr__``/``__setattr__`` 透明代理访问。
    实现 SimEngine 协议，返回统一的 TickSnapshot 和 EngineSummary。

    Attributes:
        neurons: 神经元字典，键为神经元 ID。
        queue: 事件优先队列。
        replay: tick 快照历史记录。
        expected_state: 预测状态（用于计算预测误差）。
        corpses: 当前存活的尸体列表。
        corpse_pool: 尸体对象池，复用实例减少分配。

    Example:
        >>> net = Network(neurons={"n1": Neuron(id="n1")})
        >>> snapshot = net.tick(tick=1, dt=1.0)
    """

    neurons: dict[str, Neuron]
    queue: EventQueue = field(default_factory=EventQueue)
    replay: list[dict[str, object]] = field(default_factory=list)
    expected_state: dict[str, float] | None = None
    corpses: list[Corpse] = field(default_factory=list)
    # 以下字段在 __post_init__ 中转移到组件，保留声明以兼容 dataclass 构造签名
    reproduction_threshold: float = 2.5
    reproduction_share: float = 0.4
    mutation_scale: float = 0.05
    energy_decay: float = 0.003
    max_neurons: int = 50
    max_links: int = 200
    corpse_ttl: int = 50
    scavenger_radius: float = 2.5
    input_current_scale: float = 1.5
    background_current: float = 0.0
    spontaneous_spike_rate: float = 0.0
    move_cost: float = 0.01
    energy_cap: float = 5.0
    adaptive_decay: bool = True
    adaptive_reproduction: bool = True
    endangered_threshold: int = 10
    fitness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "propagation": 1.0,
            "energy_efficiency": 0.5,
            "prediction_hit": 0.8,
        }
    )
    congestion_penalty: float = 0.1
    min_neurons: int = 2
    evolution_mutation_amplitude: float | None = None
    energy_budget_per_tick: float = 0.0
    birth_count: int = 0
    death_count: int = 0
    total_spikes: int = 0
    total_propagations: int = 0
    current_tick: int = 0
    _pe_history: list[float] = field(default_factory=list)
    _endangered_mode: bool = False
    schema_version: str = SCHEMA_VERSION
    corpse_pool: CorpsePool = field(default_factory=CorpsePool)
    corpse_pool_max_size: int = 2048

    def __post_init__(self) -> None:
        # 构建配置组件
        config_kwargs = {}
        for fname in _NETWORK_CONFIG_FIELDS:
            if fname in self.__dict__:
                config_kwargs[fname] = self.__dict__[fname]
        object.__setattr__(self, "_config", NetworkConfig(**config_kwargs))

        # 构建状态组件
        state_kwargs = {}
        for fname in _NETWORK_STATE_FIELDS:
            if fname in self.__dict__:
                state_kwargs[fname] = self.__dict__[fname]
        object.__setattr__(self, "_state", NetworkState(**state_kwargs))

        # 从 __dict__ 中删除已转移到组件的字段
        for fname in _NETWORK_CONFIG_FIELDS | _NETWORK_STATE_FIELDS:
            self.__dict__.pop(fname, None)

    def __getattr__(self, name: str) -> object:
        if name in _NETWORK_CONFIG_FIELDS:
            return getattr(self._config, name)
        if name in _NETWORK_STATE_FIELDS:
            return getattr(self._state, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: object) -> None:
        # 组件初始化前回退到 object.__setattr__
        if "_config" not in self.__dict__ or "_state" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        if name in _NETWORK_CONFIG_FIELDS:
            setattr(self._config, name, value)
        elif name in _NETWORK_STATE_FIELDS:
            setattr(self._state, name, value)
        else:
            object.__setattr__(self, name, value)

    def __setstate__(self, state: dict) -> None:
        """pickle 反序列化兼容 — 处理旧格式（无 _config/_state）"""
        self.__dict__.update(state)
        if "_config" not in state or "_state" not in state:
            # 旧 checkpoint：从扁平字段重建组件
            config_kwargs = {}
            for fname in _NETWORK_CONFIG_FIELDS:
                if fname in self.__dict__:
                    config_kwargs[fname] = self.__dict__[fname]
            object.__setattr__(self, "_config", NetworkConfig(**config_kwargs))

            state_kwargs = {}
            for fname in _NETWORK_STATE_FIELDS:
                if fname in self.__dict__:
                    state_kwargs[fname] = self.__dict__[fname]
            object.__setattr__(self, "_state", NetworkState(**state_kwargs))

            for fname in _NETWORK_CONFIG_FIELDS | _NETWORK_STATE_FIELDS:
                self.__dict__.pop(fname, None)

    def connect(
        self, source_id: str, target_id: str, weight: float = 1.0, delay: int = 0
    ) -> None:
        """在两个神经元之间建立单向连接。

        Args:
            source_id: 源神经元 ID。
            target_id: 目标神经元 ID。
            weight: 连接权重，>= 0，默认 1.0。
            delay: 传播延迟（tick 数），>= 0，默认 0。
        """
        if delay < 0:
            delay = 0
        neuron = self.neurons[source_id]
        neuron.links[target_id] = {"weight": max(0.0, weight), "delay": delay}

    def connect_by_proximity(
        self, radius: float = 5.0, max_links_per_neuron: int = 5
    ) -> None:
        """基于空间距离自动建立连接，权重与距离成反比。

        Args:
            radius: 连接搜索半径，默认 5.0。
            max_links_per_neuron: 每个神经元最大新增连接数，默认 5。
        """
        alive = [n for n in self.neurons.values() if n.alive]
        if not alive:
            return

        # 构建空间网格加速查找
        cell_size = radius if radius > 0 else 1.0
        grid: dict[tuple[int, int], list[Neuron]] = {}
        for n in alive:
            key = (int(n.pos[0] // cell_size), int(n.pos[1] // cell_size))
            grid.setdefault(key, []).append(n)

        for source in alive:
            cx, cy = int(source.pos[0] // cell_size), int(source.pos[1] // cell_size)
            candidates: list[tuple[Neuron, float]] = []

            # 搜索相邻网格
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for target in grid.get((cx + dx, cy + dy), []):
                        if source.id == target.id:
                            continue

                        # 计算欧氏距离
                        ddx = target.pos[0] - source.pos[0]
                        ddy = target.pos[1] - source.pos[1]
                        dist = math.sqrt(ddx * ddx + ddy * ddy)

                        if dist <= radius:
                            candidates.append((target, dist))

            # 按距离排序，取最近的 N 个
            candidates.sort(key=lambda x: x[1])

            # 建立连接，权重与距离成反比，延迟与距离成正比
            links_added = 0
            for target, dist in candidates:
                if links_added >= max_links_per_neuron:
                    break
                if target.id not in source.links:
                    # 距离越近，权重越大 (0.1 ~ 1.0)
                    weight = max(0.1, 1.0 - (dist / radius))
                    # 距离越远，延迟越大 (1 ~ 5)
                    delay = max(1, int((dist / radius) * 5))
                    self.connect(source.id, target.id, weight=weight, delay=delay)
                    links_added += 1

    def add_neuron(self, neuron: Neuron) -> None:
        """向网络中添加一个神经元。

        Args:
            neuron: 待添加的神经元实例。
        """
        self.neurons[neuron.id] = neuron

    def enqueue(self, events: Iterable[Event]) -> None:
        """将外部事件批量注入事件队列。

        Args:
            events: 待注入的事件可迭代对象。
        """
        self.queue.push_many(events)

    def tick(
        self, tick: int, dt: float = 1.0, run_evolution: bool = True
    ) -> TickSnapshot:
        """执行一个仿真时间步 — 事件处理、动力学更新、演化。

        Args:
            tick: 当前时间步编号。
            dt: 时间步长，单位 ms，默认 1.0。
            run_evolution: 是否执行演化步骤（繁殖/死亡），默认 True。

        Returns:
            本 tick 的快照字典，包含放电数、存活数、预测误差等。
        """
        ready, inputs, effective_decay, budget_share = self._process_inputs(tick, dt)
        emitted, tick_spikes = self._update_dynamics(
            tick=tick,
            dt=dt,
            inputs=inputs,
            effective_decay=effective_decay,
            budget_share=budget_share,
        )
        self.queue.push_many(emitted)
        self.total_spikes += tick_spikes
        self.total_propagations += len(emitted)
        snapshot = self._collect_tick_stats(
            tick=tick,
            ready_count=len(ready),
            emitted_count=len(emitted),
            tick_spikes=tick_spikes,
        )

        if run_evolution:
            births, deaths = self._run_evolution_step()
            snapshot["births"] = births
            snapshot["deaths"] = deaths

        return snapshot

    def _process_inputs(
        self, tick: int, dt: float
    ) -> tuple[list[Event], dict[str, list[Event]], float, float]:
        _ = dt
        self.current_tick = tick
        ready = self.queue.pop_ready(tick)
        inputs: dict[str, list[Event]] = defaultdict(list)
        for event in ready:
            if event.src_id not in self.neurons:
                continue
            inputs[event.src_id].append(event)

        alive_count = sum(1 for n in self.neurons.values() if n.alive)
        effective_decay = self._calculate_adaptive_decay(alive_count)
        budget_share = (
            self.energy_budget_per_tick / alive_count
            if self.energy_budget_per_tick > 0 and alive_count > 0
            else 0.0
        )

        if alive_count <= self.endangered_threshold and not self._endangered_mode:
            self._endangered_mode = True
        elif alive_count > self.endangered_threshold * 1.5:
            self._endangered_mode = False
        return ready, inputs, effective_decay, budget_share

    def _update_dynamics(
        self,
        *,
        tick: int,
        dt: float,
        inputs: dict[str, list[Event]],
        effective_decay: float,
        budget_share: float,
    ) -> tuple[list[Event], int]:
        emitted: list[Event] = []
        tick_spikes = 0
        for neuron in list(self.neurons.values()):
            neuron.start_tick()
            events = inputs.get(neuron.id, [])
            input_current = self.background_current
            for event in events:
                neuron.receive(event)
                w = float(event.meta.get("weight", 1.0))
                input_current += safe_float(w * self.input_current_scale)

            spontaneous_spike = False
            if self.spontaneous_spike_rate > 0.0:
                spontaneous_spike = random.random() < self.spontaneous_spike_rate
            if spontaneous_spike:
                neuron.on_spike()
                spiked = True
            else:
                spiked = neuron.update(dt, input_current)

            if spiked:
                tick_spikes += 1
                if neuron.last_event is not None:
                    emitted.extend(neuron.emit(neuron.last_event, current_tick=tick))
                elif spontaneous_spike:
                    emitted.extend(
                        neuron.emit(
                            Event(
                                token_id=0,
                                t=tick,
                                src_id=neuron.id,
                                meta={"spontaneous": True, "weight": 1.0},
                            ),
                            current_tick=tick,
                        )
                    )
            neuron.move()
            decay = effective_decay * (0.5 if self._endangered_mode else 1.0)
            neuron.energy = max(0.0, neuron.energy - decay)
            if budget_share > 0.0:
                neuron.energy += budget_share
            if neuron.energy > self.energy_cap:
                neuron.energy = self.energy_cap
            energy_loss = max(0.0, neuron.last_energy - neuron.energy)
            reward = calculate_reward(neuron.states["novelty"], energy_loss)
            neuron.apply_reward(reward)
            neuron.finalize_tick(had_input=bool(events))
        return emitted, tick_spikes

    def _collect_tick_stats(
        self, *, tick: int, ready_count: int, emitted_count: int, tick_spikes: int
    ) -> TickSnapshot:
        actual_state = self._aggregate_state()
        prediction_error = self._update_prediction_error(actual_state)
        self._pe_history.append(prediction_error)

        alive_neurons = [neuron for neuron in self.neurons.values() if neuron.alive]
        alive_count = len(alive_neurons)
        mean_energy = (
            sum(n.energy for n in alive_neurons) / alive_count
            if alive_count > 0
            else 0.0
        )
        snapshot: TickSnapshot = {
            "schema_version": self.schema_version,
            "tick": tick,
            "ready_events": ready_count,
            "propagation_count": emitted_count,
            "alive_neurons": alive_count,
            "total_neurons": len(self.neurons),
            "prediction_error": prediction_error,
            "tick_spikes": tick_spikes,
            "births": 0,
            "deaths": 0,
            "mean_energy": mean_energy,
        }
        self.replay.append(snapshot)
        return snapshot

    def _run_evolution_step(self) -> tuple[int, int]:
        self._update_corpses()
        births = self.check_reproduction()
        deaths = self.check_death()
        self.birth_count += births
        self.death_count += deaths
        return births, deaths

    def pe_moving_average(self, window: int = 10) -> float:
        """PE 滑动均值"""
        if not self._pe_history:
            return 0.0
        recent = self._pe_history[-window:]
        return sum(recent) / len(recent)

    def active_neuron_ratio(self) -> float:
        """活跃神经元比例"""
        if not self.neurons:
            return 0.0
        alive = sum(1 for n in self.neurons.values() if n.alive)
        return alive / len(self.neurons)

    def _aggregate_state(self) -> dict[str, float]:
        alive = [neuron for neuron in self.neurons.values() if neuron.alive]
        if not alive:
            return {"energy": 0.0, "entropy": 0.0, "emotion": 0.0, "novelty": 0.0}
        size = len(alive)
        return {
            "energy": sum(neuron.states["energy"] for neuron in alive) / size,
            "entropy": sum(neuron.states["entropy"] for neuron in alive) / size,
            "emotion": sum(neuron.states["emotion"] for neuron in alive) / size,
            "novelty": sum(neuron.states["novelty"] for neuron in alive) / size,
        }

    def _update_prediction_error(self, actual_state: dict[str, float]) -> float:
        if self.expected_state is None:
            self.expected_state = dict(actual_state)
            return 0.0
        prediction_error = math.sqrt(
            sum(
                (actual_state[key] - self.expected_state.get(key, 0.0)) ** 2
                for key in actual_state
            )
        )
        prediction_error = safe_float(prediction_error)
        alpha = 0.3
        self.expected_state = {
            key: self.expected_state.get(key, 0.0)
            + alpha * (actual_state[key] - self.expected_state.get(key, 0.0))
            for key in actual_state
        }
        return prediction_error

    def check_reproduction(self) -> int:
        """执行繁殖检查 — 高适应度且能量充足的神经元产生子代。

        Returns:
            本轮新增的子代数量。
        """
        if len(self.neurons) >= self.max_neurons:
            return 0
        candidates = [neuron for neuron in self.neurons.values() if neuron.alive]
        # 按 fitness 排序 — 高适应度优先繁殖
        candidates.sort(key=lambda n: n.fitness(self.fitness_weights), reverse=True)

        # 计算自适应繁殖成本
        alive_count = len(candidates)
        effective_share = self._calculate_adaptive_reproduction_share(alive_count)
        effective_mutation_scale = (
            self.evolution_mutation_amplitude
            if self.evolution_mutation_amplitude is not None
            else self.mutation_scale
        )
        births = 0

        for neuron in candidates:
            if len(self.neurons) >= self.max_neurons:
                break
            if neuron.energy < self.reproduction_threshold:
                continue
            # 拥塞惩罚: 该神经元连接过多时降低繁殖概率
            link_count = len(neuron.links)
            if link_count > 10 and random.random() < self.congestion_penalty:
                continue
            child = neuron.spawn(max(0.0, effective_mutation_scale), effective_share)
            neuron.energy = max(0.0, neuron.energy - child.energy)
            self.add_neuron(child)
            self.connect(neuron.id, child.id, weight=1.0, delay=1)
            self.connect(child.id, neuron.id, weight=1.0, delay=1)
            births += 1
        self._prune_links()
        return births

    def _calculate_adaptive_decay(self, alive_count: int) -> float:
        """根据种群密度计算自适应衰减率"""
        if not self.adaptive_decay:
            return self.energy_decay

        # 种群密度比例
        density_ratio = alive_count / self.max_neurons if self.max_neurons > 0 else 0.5

        # 密度高时增加衰减，密度低时减少衰减
        if density_ratio > DENSITY_HIGH_THRESHOLD:
            # 过度拥挤，增加衰减
            return self.energy_decay * DECAY_HIGH_MULTIPLIER
        elif density_ratio < DENSITY_LOW_THRESHOLD:
            # 种群稀少，减少衰减
            return self.energy_decay * DECAY_LOW_MULTIPLIER
        else:
            return self.energy_decay

    def _calculate_adaptive_reproduction_share(self, alive_count: int) -> float:
        """根据种群规模计算自适应繁殖成本"""
        if not self.adaptive_reproduction:
            return self.reproduction_share

        # 种群规模比例
        population_ratio = (
            alive_count / self.max_neurons if self.max_neurons > 0 else 0.5
        )

        # 接近上限时增加繁殖成本，远离上限时降低成本
        if population_ratio > POPULATION_HIGH_THRESHOLD:
            # 接近容量上限，增加成本（母体分享更多能量）
            return min(REPRO_SHARE_MAX, self.reproduction_share * REPRO_SHARE_HIGH_MULTIPLIER)
        elif population_ratio < POPULATION_LOW_THRESHOLD:
            # 种群稀少，降低成本（鼓励繁殖）
            return max(REPRO_SHARE_MIN, self.reproduction_share * REPRO_SHARE_LOW_MULTIPLIER)
        else:
            return self.reproduction_share

    def check_death(self) -> int:
        """执行死亡检查 — 移除能量耗尽的神经元，生成尸体。

        当存活数低于 min_neurons 时启动保护机制，复活能量最高的死亡神经元。

        Returns:
            本轮死亡的神经元数量。
        """
        if len(self.neurons) > self.max_neurons:
            ordered = sorted(self.neurons.values(), key=lambda item: item.energy)
            overflow = len(self.neurons) - self.max_neurons
            for neuron in ordered[:overflow]:
                neuron.alive = False

        dead_ids = [
            neuron.id
            for neuron in self.neurons.values()
            if not neuron.alive or neuron.energy <= 0.0
        ]
        # 保护 min_neurons 下限
        alive_count = len(self.neurons) - len(dead_ids)
        if alive_count < self.min_neurons:
            _logger.warning(
                "种群触及 min_neurons 下限: alive=%d, dead=%d, min=%d — 启动保护机制",
                alive_count,
                len(dead_ids),
                self.min_neurons,
            )
            # 保留能量最高的死亡神经元
            dead_neurons = [
                (nid, self.neurons[nid].energy)
                for nid in dead_ids
                if nid in self.neurons
            ]
            dead_neurons.sort(key=lambda x: x[1], reverse=True)
            save_count = self.min_neurons - alive_count
            saved = {nid for nid, _ in dead_neurons[:save_count]}
            for nid in saved:
                self.neurons[nid].alive = True
                self.neurons[nid].energy = max(RESCUED_NEURON_ENERGY, self.neurons[nid].energy)
            dead_ids = [nid for nid in dead_ids if nid not in saved]

        deaths = 0
        for neuron_id in dead_ids:
            neuron_obj: Neuron | None = self.neurons.get(neuron_id)
            if neuron_obj is None:
                continue
            params = {
                "tau_m": neuron_obj.tau_m,
                "r_m": neuron_obj.r_m,
                "v_th": neuron_obj.v_th,
                "v_rest": neuron_obj.v_rest,
                "v_reset": neuron_obj.v_reset,
                "spike_cost": neuron_obj.spike_cost,
                "input_gain": neuron_obj.input_gain,
            }
            for attr in ["a", "b", "c", "d", "v_peak", "delta_t", "tau_w"]:
                if hasattr(neuron_obj, attr):
                    value = getattr(neuron_obj, attr)
                    if isinstance(value, (int, float)):
                        params[attr] = float(value)
            self.corpses.append(
                self.corpse_pool.acquire(
                    pos=neuron_obj.pos,
                    energy=neuron_obj.energy,
                    params=params,
                    ttl=self.corpse_ttl,
                )
            )
            self.neurons.pop(neuron_id, None)
            deaths += 1
        return deaths

    def _update_corpses(self) -> None:
        if not self.corpses:
            return
        alive = [neuron for neuron in self.neurons.values() if neuron.alive]
        updated: list[Corpse] = []

        # 构建简单网格索引加速近邻搜索
        cell_size = self.scavenger_radius if self.scavenger_radius > 0 else 1.0
        grid: dict[tuple[int, int], list[Neuron]] = {}
        for n in alive:
            key = (int(n.pos[0] // cell_size), int(n.pos[1] // cell_size))
            grid.setdefault(key, []).append(n)

        for corpse in self.corpses:
            corpse.ttl -= 1
            # 能量衰减
            corpse.energy = max(0.0, corpse.energy - corpse.decay_rate)
            if corpse.energy > 0 and alive:
                cx, cy = int(corpse.pos[0] // cell_size), int(
                    corpse.pos[1] // cell_size
                )
                nearest = None
                nearest_dist = None
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for n in grid.get((cx + dx, cy + dy), []):
                            ddx = n.pos[0] - corpse.pos[0]
                            ddy = n.pos[1] - corpse.pos[1]
                            dist = math.sqrt(ddx * ddx + ddy * ddy)
                            if dist <= self.scavenger_radius and (
                                nearest_dist is None or dist < nearest_dist
                            ):
                                nearest, nearest_dist = n, dist
                if nearest is not None:
                    nearest.energy += corpse.energy
                    corpse.energy = 0.0
            if corpse.ttl > 0 and corpse.energy > 0.0:
                updated.append(corpse)
            else:
                self.corpse_pool.release(corpse, self.corpse_pool_max_size)
        self.corpses = updated

    def _prune_links(self) -> None:
        total_links = sum(len(neuron.links) for neuron in self.neurons.values())
        if total_links <= self.max_links:
            return
        all_links: list[tuple[str, str]] = []
        for neuron_id, neuron in self.neurons.items():
            for target_id in neuron.links.keys():
                all_links.append((neuron_id, target_id))
        random.shuffle(all_links)
        to_remove = total_links - self.max_links
        for source_id, target_id in all_links[:to_remove]:
            if source_id in self.neurons:
                self.neurons[source_id].links.pop(target_id, None)

    def save_checkpoint(self, path: str | Path, tick: int | None = None) -> Path:
        """保存网络 checkpoint（pickle 格式）。

        Args:
            path: 输出文件路径。
            tick: 保存的 tick 编号，None 时使用 current_tick。

        Returns:
            实际写入的文件路径。

        Raises:
            OSError: 文件写入失败时抛出。
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_tick = self.current_tick if tick is None else int(tick)
        payload = {
            "schema_version": self.schema_version,
            "tick": save_tick,
            "random_state": random.getstate(),
            "network": self,
        }
        with checkpoint_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return checkpoint_path

    @staticmethod
    def load_checkpoint(path: str | Path) -> "Network":
        """从 checkpoint 恢复网络，并恢复随机数状态。

        Args:
            path: checkpoint 文件路径。

        Returns:
            恢复后的 Network 实例。

        Raises:
            ValueError: checkpoint 格式非法或 schema 主版本不兼容时抛出。
        """
        checkpoint_path = Path(path)
        with checkpoint_path.open("rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict):
            raise ValueError("checkpoint 格式非法：顶层必须为 dict")

        checkpoint_schema = str(payload.get("schema_version", ""))
        if checkpoint_schema:
            current_major = str(SCHEMA_VERSION).split(".", maxsplit=1)[0]
            loaded_major = checkpoint_schema.split(".", maxsplit=1)[0]
            if loaded_major != current_major:
                raise ValueError(
                    f"checkpoint schema 主版本不兼容: {checkpoint_schema} != {SCHEMA_VERSION}"
                )

        network = payload.get("network")
        if not isinstance(network, Network):
            raise ValueError("checkpoint 格式非法：缺少 network 对象")

        restored_tick = int(payload.get("tick", getattr(network, "current_tick", 0)))
        network.current_tick = max(0, restored_tick)

        random_state = payload.get("random_state")
        if random_state is not None:
            random.setstate(random_state)

        return network

    def get_summary(self) -> EngineSummary:
        """生成网络运行摘要。

        Returns:
            包含存活数、放电数、能量均值等统计信息的字典。
        """
        alive = [n for n in self.neurons.values() if n.alive]
        return {
            "schema_version": self.schema_version,
            "total_neurons": len(self.neurons),
            "alive_neurons": len(alive),
            "total_spikes": self.total_spikes,
            "total_propagations": self.total_propagations,
            "birth_count": self.birth_count,
            "death_count": self.death_count,
            "corpse_pool_free": len(self.corpse_pool.free),
            "corpse_pool_reused": self.corpse_pool.reused_count,
            "pe_moving_avg": self.pe_moving_average(),
            "active_ratio": self.active_neuron_ratio(),
            "mean_energy": sum(n.energy for n in alive) / len(alive) if alive else 0.0,
        }


# 删除 dataclass 在类上生成的默认值属性，避免遮蔽 __getattr__ 路由
for _fname in _NETWORK_CONFIG_FIELDS | _NETWORK_STATE_FIELDS:
    if _fname in Network.__dict__:
        delattr(Network, _fname)
