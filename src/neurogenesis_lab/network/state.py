"""网络配置与运行时状态 — network 子包的组件字段部分。

``NetworkConfig`` / ``NetworkState`` 由 ``Network`` 通过 ``ComponentMixin``
透明代理访问；``_NETWORK_CONFIG_FIELDS`` / ``_NETWORK_STATE_FIELDS`` 用于
``__getattr__`` / ``__setattr__`` 的字段路由。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neurogenesis_lab.config import SCHEMA_VERSION


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
_NETWORK_CONFIG_FIELDS = frozenset(f.name for f in NetworkConfig.__dataclass_fields__.values())
_NETWORK_STATE_FIELDS = frozenset(f.name for f in NetworkState.__dataclass_fields__.values())
