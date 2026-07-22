"""Network 属性类型基类 — 供 Mixin 静态类型检查使用。

``_NetworkAttrs`` 仅声明 ``Network`` 实例属性的类型（裸注解，不创建类属性，
因此不会遮蔽 ``ComponentMixin.__getattr__`` 的字段路由），并提供若干被多个
Mixin 共用的具体方法（``connect`` / ``add_neuron`` / 自适应委托）。

所有 Network 功能 Mixin 均继承本类以获得 ``self.*`` 属性的类型信息；
运行时这些属性由 ``Network`` 的 dataclass 字段或 ``ComponentMixin`` 路由提供。
"""

from __future__ import annotations

from neurogenesis_lab.adaptive import (
    calculate_adaptive_decay,
    calculate_adaptive_reproduction_share,
)
from neurogenesis_lab.network.corpse import Corpse, CorpsePool
from neurogenesis_lab.network.event_queue import EventQueue
from neurogenesis_lab.neuron import Neuron
from neurogenesis_lab.protocols import TickSnapshot


class _NetworkAttrs:
    """Network 实例属性的类型声明 + 跨 Mixin 共用方法。"""

    # ── 直接实例字段 ──────────────────────────────────────
    neurons: dict[str, Neuron]
    queue: EventQueue
    replay: list[TickSnapshot]
    expected_state: dict[str, float] | None
    corpses: list[Corpse]
    corpse_pool: CorpsePool

    # ── 配置字段（运行时经 ComponentMixin 路由到 _config） ──
    reproduction_threshold: float
    reproduction_share: float
    mutation_scale: float
    energy_decay: float
    max_neurons: int
    max_links: int
    corpse_ttl: int
    scavenger_radius: float
    input_current_scale: float
    background_current: float
    spontaneous_spike_rate: float
    move_cost: float
    energy_cap: float
    adaptive_decay: bool
    adaptive_reproduction: bool
    endangered_threshold: int
    fitness_weights: dict[str, float]
    congestion_penalty: float
    min_neurons: int
    evolution_mutation_amplitude: float | None
    energy_budget_per_tick: float
    schema_version: str
    corpse_pool_max_size: int

    # ── 状态字段（运行时经 ComponentMixin 路由到 _state） ──
    birth_count: int
    death_count: int
    total_spikes: int
    total_propagations: int
    current_tick: int
    _pe_history: list[float]
    _endangered_mode: bool

    # ── 跨 Mixin 共用方法 ─────────────────────────────────
    def connect(self, source_id: str, target_id: str, weight: float = 1.0, delay: int = 0) -> None:
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

    def add_neuron(self, neuron: Neuron) -> None:
        """向网络中添加一个神经元。

        Args:
            neuron: 待添加的神经元实例。
        """
        self.neurons[neuron.id] = neuron

    def _calculate_adaptive_decay(self, alive_count: int) -> float:
        """计算自适应能量衰减 — 委托 adaptive 模块（保留向后兼容 API）。

        Args:
            alive_count: 当前存活神经元数量。

        Returns:
            根据种群密度调整后的能量衰减值。
        """
        return calculate_adaptive_decay(
            self.adaptive_decay, self.energy_decay, alive_count, self.max_neurons
        )

    def _calculate_adaptive_reproduction_share(self, alive_count: int) -> float:
        """计算自适应繁殖份额 — 委托 adaptive 模块（保留向后兼容 API）。

        Args:
            alive_count: 当前存活神经元数量。

        Returns:
            根据种群规模调整后的繁殖能量分享比例。
        """
        return calculate_adaptive_reproduction_share(
            self.adaptive_reproduction,
            self.reproduction_share,
            alive_count,
            self.max_neurons,
        )
