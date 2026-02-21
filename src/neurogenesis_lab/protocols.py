"""双引擎统一接口协议 — TickSnapshot / EngineSummary / SimEngine"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class TickSnapshot(TypedDict, total=False):
    """tick() 方法返回的统一快照类型"""

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
    # 学习引擎指标（可选，由 AutoLearner 填充）
    rule_count: int
    prediction_accuracy: float


class EngineSummary(TypedDict, total=False):
    """get_summary() 方法返回的统一摘要类型"""

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
    # 学习引擎指标（可选）
    prediction_accuracy: float
    rule_count: int


@runtime_checkable
class SimEngine(Protocol):
    """仿真引擎协议 — Network 和 VectorizedNetwork 均需满足"""

    schema_version: str
    total_spikes: int
    birth_count: int
    death_count: int

    def tick(self, tick: int, dt: float = ..., **kwargs) -> TickSnapshot: ...
    def get_summary(self) -> EngineSummary: ...
