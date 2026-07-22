"""神经网络仿真引擎 — network 子包的主类。

``Network`` 通过多个功能 Mixin（dynamics / lifecycle / persistence）组合而成，
配置与状态字段经 ``ComponentMixin`` 透明代理到 ``NetworkConfig`` / ``NetworkState``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, Iterable

from neurogenesis_lab._components import ComponentMixin
from neurogenesis_lab.config import (
    PROXIMITY_DELAY_SCALE,
    PROXIMITY_WEIGHT_MIN,
    SCHEMA_VERSION,
)
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.network.corpse import Corpse, CorpsePool
from neurogenesis_lab.network.dynamics import NetworkDynamicsMixin
from neurogenesis_lab.network.event_queue import EventQueue
from neurogenesis_lab.network.lifecycle import NetworkLifecycleMixin
from neurogenesis_lab.network.persistence import NetworkPersistenceMixin
from neurogenesis_lab.network.state import (
    _NETWORK_CONFIG_FIELDS,
    _NETWORK_STATE_FIELDS,
    NetworkConfig,
    NetworkState,
)
from neurogenesis_lab.neuron import Neuron
from neurogenesis_lab.protocols import EngineSummary, TickSnapshot


@dataclass
class Network(
    ComponentMixin,
    NetworkDynamicsMixin,
    NetworkLifecycleMixin,
    NetworkPersistenceMixin,
):
    """神经网络仿真引擎 — 管理神经元集合、事件队列和演化逻辑。

    内部配置和状态分别委托给 NetworkConfig 和 NetworkState 组件，
    通过 ``ComponentMixin`` 透明代理访问。
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

    _component_specs: ClassVar = [
        ("_config", NetworkConfig, _NETWORK_CONFIG_FIELDS),
        ("_state", NetworkState, _NETWORK_STATE_FIELDS),
    ]

    neurons: dict[str, Neuron]
    queue: EventQueue = field(default_factory=EventQueue)
    replay: list[TickSnapshot] = field(default_factory=list)
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

    def connect_by_proximity(self, radius: float = 5.0, max_links_per_neuron: int = 5) -> None:
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
                    # 距离越近，权重越大 (PROXIMITY_WEIGHT_MIN ~ 1.0)
                    weight = max(PROXIMITY_WEIGHT_MIN, 1.0 - (dist / radius))
                    # 距离越远，延迟越大 (1 ~ PROXIMITY_DELAY_SCALE)
                    delay = max(1, int((dist / radius) * PROXIMITY_DELAY_SCALE))
                    self.connect(source.id, target.id, weight=weight, delay=delay)
                    links_added += 1

    def enqueue(self, events: Iterable[Event]) -> None:
        """将外部事件批量注入事件队列。

        Args:
            events: 待注入的事件可迭代对象。
        """
        self.queue.push_many(events)

    def tick(self, tick: int, dt: float = 1.0, run_evolution: bool = True) -> TickSnapshot:
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
ComponentMixin.cleanup_dataclass_defaults(Network)
