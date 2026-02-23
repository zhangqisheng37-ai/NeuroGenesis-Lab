"""集中配置系统 — 参数可控化 + 随机种子统一入口"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "2.0.0"

# ── 共享常量 ──────────────────────────────────────────────

# 自适应衰减
DENSITY_HIGH_THRESHOLD = 0.8
DENSITY_LOW_THRESHOLD = 0.3
DECAY_HIGH_MULTIPLIER = 1.2
DECAY_LOW_MULTIPLIER = 0.7

# 自适应繁殖
POPULATION_HIGH_THRESHOLD = 0.8
POPULATION_LOW_THRESHOLD = 0.3
REPRO_SHARE_HIGH_MULTIPLIER = 1.3
REPRO_SHARE_LOW_MULTIPLIER = 0.7
REPRO_SHARE_MAX = 0.6
REPRO_SHARE_MIN = 0.2

# 神经元保护
RESCUED_NEURON_ENERGY = 0.1

# 遗传属性
HERITABLE_DYNAMICS_ATTRS = (
    "tau_m", "r_m", "v_th", "v_rest", "v_reset",
    "spike_cost", "input_gain", "move_cost",
)
HERITABLE_SUBCLASS_ATTRS = ("a", "b", "c", "d", "v_peak", "delta_t", "tau_w")
HERITABLE_ATTRS = HERITABLE_DYNAMICS_ATTRS + HERITABLE_SUBCLASS_ATTRS

# 子代位置抖动
SPAWN_POSITION_JITTER = 0.5

# 向量化引擎移动抖动
MOVE_JITTER_RANGE = 1.0


@dataclass(frozen=True)
class Config:
    """向后兼容的最小配置。

    Attributes:
        ticks: 仿真总步数，> 0，默认 10。
        network_size: 初始网络规模（神经元数），> 0，默认 16。
        energy_decay: 每 tick 能量衰减量，>= 0，默认 0.01。
        seed: 随机种子，默认 42。
    """

    ticks: int = 10
    network_size: int = 16
    energy_decay: float = 0.01
    seed: int = 42


def default_config() -> Config:
    """返回默认的最小配置实例。

    Returns:
        使用默认参数的 Config 实例。
    """
    return Config()


@dataclass
class NeuronParams:
    """神经元动力学参数。

    Attributes:
        v_rest: 静息电位，单位 mV，默认 -65.0。
        v_reset: 放电后重置电位，单位 mV，默认 -65.0。
        v_th: 放电阈值，单位 mV，默认 -55.0。
        tau_m: 膜时间常数，单位 ms，必须 > 0，默认 20.0。
        r_m: 膜电阻缩放因子，无量纲，默认 1.0。
        spike_cost: 每次放电能量消耗，>= 0，默认 0.05。
        input_gain: 接收事件时的能量增益，>= 0，默认 0.05。
        mem_size: 记忆缓冲区容量（事件条数），> 0，默认 64。
    """

    v_rest: float = -65.0
    v_reset: float = -65.0
    v_th: float = -55.0
    tau_m: float = 20.0
    r_m: float = 1.0
    spike_cost: float = 0.05
    input_gain: float = 0.05
    mem_size: int = 64

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.v_rest):
            errors.append(f"v_rest 不是有限数: {self.v_rest}")
        if not math.isfinite(self.v_reset):
            errors.append(f"v_reset 不是有限数: {self.v_reset}")
        if not math.isfinite(self.v_th):
            errors.append(f"v_th 不是有限数: {self.v_th}")
        if self.tau_m <= 0:
            errors.append(f"tau_m 必须为正: {self.tau_m}")
        if self.spike_cost < 0:
            errors.append(f"spike_cost 不能为负: {self.spike_cost}")
        if self.input_gain < 0:
            errors.append(f"input_gain 不能为负: {self.input_gain}")
        if self.mem_size <= 0:
            errors.append(f"mem_size 必须为正: {self.mem_size}")
        return errors


@dataclass
class NetworkParams:
    """网络级参数。

    Attributes:
        reproduction_threshold: 繁殖能量阈值，> 0，默认 2.5。
        reproduction_share: 繁殖能量分配比例，范围 (0, 1]，默认 0.4。
        mutation_scale: 变异幅度缩放因子，>= 0，默认 0.05。
        energy_decay: 每 tick 基础能量衰减量，>= 0，默认 0.003。
        max_neurons: 最大神经元数量，> 0，默认 50。
        max_links: 最大连接数量，> 0，默认 200。
        corpse_ttl: 尸体存活 tick 数，> 0，默认 50。
        scavenger_radius: 拾荒搜索半径，>= 0，默认 2.5。
        input_current_scale: 输入电流缩放因子，>= 0，默认 1.5。
        background_current: 背景电流，>= 0，默认 0.0。
        spontaneous_spike_rate: 自发放电概率，范围 [0, 1]，默认 0.0。
        low_activation_threshold: 低激活阈值，>= 0，默认 0。
        move_cost: 每次移动能量消耗，>= 0，默认 0.01。
        energy_cap: 能量上限，> 0，默认 5.0。
        adaptive_decay: 是否启用自适应衰减，默认 True。
        adaptive_reproduction: 是否启用自适应繁殖成本，默认 True。
        endangered_threshold: 濒危告警阈值（存活数），>= 0，默认 10。
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
    low_activation_threshold: int = 0
    move_cost: float = 0.01
    # 长程稳定性参数
    energy_cap: float = 5.0  # 能量上限，防止无限积累
    adaptive_decay: bool = True  # 是否启用自适应衰减
    adaptive_reproduction: bool = True  # 是否启用自适应繁殖成本
    endangered_threshold: int = 10  # 濒危告警阈值

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.reproduction_threshold <= 0:
            errors.append(
                f"reproduction_threshold 必须为正: {self.reproduction_threshold}"
            )
        if not (0 < self.reproduction_share <= 1):
            errors.append(
                f"reproduction_share 必须在 (0, 1]: {self.reproduction_share}"
            )
        if self.mutation_scale < 0:
            errors.append(f"mutation_scale 不能为负: {self.mutation_scale}")
        if self.energy_decay < 0:
            errors.append(f"energy_decay 不能为负: {self.energy_decay}")
        if self.max_neurons <= 0:
            errors.append(f"max_neurons 必须为正: {self.max_neurons}")
        if self.max_links <= 0:
            errors.append(f"max_links 必须为正: {self.max_links}")
        if self.corpse_ttl <= 0:
            errors.append(f"corpse_ttl 必须为正: {self.corpse_ttl}")
        if self.scavenger_radius < 0:
            errors.append(f"scavenger_radius 不能为负: {self.scavenger_radius}")
        if self.input_current_scale < 0:
            errors.append(f"input_current_scale 不能为负: {self.input_current_scale}")
        if self.background_current < 0:
            errors.append(f"background_current 不能为负: {self.background_current}")
        if not (0.0 <= self.spontaneous_spike_rate <= 1.0):
            errors.append(
                f"spontaneous_spike_rate 必须在 [0, 1] 范围内: {self.spontaneous_spike_rate}"
            )
        if self.low_activation_threshold < 0:
            errors.append(
                f"low_activation_threshold 不能为负: {self.low_activation_threshold}"
            )
        if self.move_cost < 0:
            errors.append(f"move_cost 不能为负: {self.move_cost}")
        if self.energy_cap <= 0:
            errors.append(f"energy_cap 必须为正: {self.energy_cap}")
        if self.endangered_threshold < 0:
            errors.append(f"endangered_threshold 不能为负: {self.endangered_threshold}")
        return errors


@dataclass
class EvolutionParams:
    """演化系统参数。

    Attributes:
        fitness_weights: 适应度各维度权重字典，键为 propagation/energy_efficiency/prediction_hit。
        mutation_amplitude: 演化变异幅度，>= 0，默认 0.05。
        energy_budget_per_tick: 每 tick 全局能量预算，0 表示不限，默认 0.0。
        congestion_penalty: 拥塞惩罚概率，范围 [0, 1]，默认 0.1。
        min_neurons: 最小存活神经元数（保护下限），>= 0，默认 2。
    """

    fitness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "propagation": 1.0,
            "energy_efficiency": 0.5,
            "prediction_hit": 0.8,
        }
    )
    mutation_amplitude: float = 0.05
    energy_budget_per_tick: float = 0.0  # 0 = 不限
    congestion_penalty: float = 0.1
    min_neurons: int = 2

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.mutation_amplitude < 0:
            errors.append(f"mutation_amplitude 不能为负: {self.mutation_amplitude}")
        if self.min_neurons < 0:
            errors.append(f"min_neurons 不能为负: {self.min_neurons}")
        return errors


@dataclass
class SimConfig:
    """仿真总配置 — 统一入口，组合所有子配置。

    Attributes:
        ticks: 仿真总步数，> 0，默认 100。
        seed: 随机种子，默认 42。
        dt: 时间步长，单位 ms，> 0，默认 1.0。
        neuron: 神经元参数。
        network: 网络参数。
        evolution: 演化参数。
        schema_version: 数据格式版本号。

    Example:
        >>> config = SimConfig(ticks=200, seed=123)
        >>> errors = config.validate()
    """

    ticks: int = 100
    seed: int = 42
    dt: float = 1.0
    neuron: NeuronParams = field(default_factory=NeuronParams)
    network: NetworkParams = field(default_factory=NetworkParams)
    evolution: EvolutionParams = field(default_factory=EvolutionParams)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ticks <= 0:
            errors.append(f"ticks 必须为正: {self.ticks}")
        if self.dt <= 0:
            errors.append(f"dt 必须为正: {self.dt}")
        if not math.isfinite(self.dt):
            errors.append(f"dt 不是有限数: {self.dt}")
        errors.extend(self.neuron.validate())
        errors.extend(self.network.validate())
        errors.extend(self.evolution.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于 jsonl 记录）"""
        return asdict(self)


def default_sim_config() -> SimConfig:
    """返回默认的仿真总配置实例。

    Returns:
        使用默认参数的 SimConfig 实例。
    """
    return SimConfig()


def seed_all(seed: int) -> None:
    """统一随机种子入口 — 覆盖 random 和 numpy 的随机状态。

    Args:
        seed: 随机种子值。
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
