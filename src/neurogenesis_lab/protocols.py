"""双引擎统一接口协议 — TickSnapshot / EngineSummary / SimEngine"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class TickSnapshot(TypedDict, total=False):
    """tick() 方法返回的统一快照类型

    核心字段由 Network.tick() / VectorizedNetwork.tick() 填充；
    rule_count / prediction_accuracy 为可选字段，由 AutoLearner 在 tick 后注入。
    """

    schema_version: str
    tick: int
    alive_neurons: int
    total_neurons: int
    tick_spikes: int
    propagation_count: int
    births: int
    deaths: int
    prediction_error: float
    mean_energy: float
    # Network 引擎额外字段（非协议核心，不依赖此字段做跨引擎兼容）
    ready_events: int
    # 学习引擎指标（可选，由 AutoLearner 在 tick() 调用后注入）
    rule_count: int
    prediction_accuracy: float


class EngineSummary(TypedDict, total=False):
    """get_summary() 方法返回的统一摘要类型

    两个引擎均保证返回核心字段；
    corpse_pool_* 为 Network 专属扩展字段，VectorizedNetwork 不提供。
    rule_count / prediction_accuracy 由 AutoLearner 独立管理，不在引擎 get_summary() 中。
    """

    schema_version: str
    total_neurons: int
    alive_neurons: int
    total_spikes: int
    total_propagations: int
    birth_count: int
    death_count: int
    pe_moving_avg: float
    active_ratio: float
    mean_energy: float
    # Network 引擎专属扩展（VectorizedNetwork 不含）
    corpse_pool_free: int
    corpse_pool_reused: int


@runtime_checkable
class SimEngine(Protocol):
    """仿真引擎协议 — Network 和 VectorizedNetwork 均需满足"""

    schema_version: str
    total_spikes: int
    birth_count: int
    death_count: int

    def tick(self, tick: int, dt: float = ..., **kwargs: object) -> TickSnapshot: ...
    def get_summary(self) -> EngineSummary: ...
