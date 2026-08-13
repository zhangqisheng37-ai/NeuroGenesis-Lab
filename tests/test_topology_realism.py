"""拓扑现实场景测试 — 真实世界拓扑中的非理想结构。

覆盖现实网络中必然出现的边缘拓扑与脏输入：
- 自环（self-loop）：神经元连接自身形成自激励
- 孤立节点：无任何边的神经元在仿真中存活
- 负权重/零权重边：不应产生传播
- 重复连接：幂等覆盖而非累积
- 未知源事件：来自已移除/外部神经元的残留事件
- NaN 权重：脏数据不应污染网络状态（safe_float 兜底）
"""

from __future__ import annotations

import math
import random

import pytest

from neurogenesis_lab.encoding import Event
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron

SENSITIVE = {"tau_m": 1.0, "r_m": 30.0, "v_th": -64.0, "move_cost": 0.0}


def _pair() -> tuple[Network, LIFNeuron, LIFNeuron]:
    """返回双神经元网络（无边），高敏感配置确保确定性放电。"""
    random.seed(1)
    n1 = LIFNeuron(energy=3.0, **SENSITIVE)
    n2 = LIFNeuron(energy=3.0, **SENSITIVE)
    net = Network(
        neurons={n1.id: n1, n2.id: n2},
        energy_decay=0.0,
        move_cost=0.0,
        input_current_scale=5.0,
        reproduction_threshold=999.0,
        max_neurons=10,
        min_neurons=0,
    )
    return net, n1, n2


def test_self_loop_connection_propagates_to_self() -> None:
    """自环边：放电后事件回到自身形成自激励，不崩溃。"""
    net, a, _ = _pair()
    net.connect(a.id, a.id, weight=1.0, delay=1)
    assert a.id in a.links, "自环连接应被建立"

    net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 40.0})])
    s1 = net.tick(1)
    assert s1["tick_spikes"] >= 1, "外部事件应触发 A 放电"

    s2 = net.tick(2)
    assert s2["ready_events"] >= 1, "A 应收到自己的传播事件（自环生效）"
    assert net.total_spikes >= 2, "自激励应再次触发放电"


def test_isolated_neuron_survives_without_links() -> None:
    """孤立节点（无入边无出边）在多 tick 仿真中保持存活且状态不受扰动。"""
    random.seed(2)
    n1 = LIFNeuron(energy=3.0, **SENSITIVE)
    n2 = LIFNeuron(energy=3.0, **SENSITIVE)
    n3 = LIFNeuron(energy=3.0, **SENSITIVE)  # 孤立节点
    net = Network(
        neurons={n1.id: n1, n2.id: n2, n3.id: n3},
        energy_decay=0.0,
        move_cost=0.0,
        input_current_scale=5.0,
        reproduction_threshold=999.0,
        max_neurons=10,
        min_neurons=0,
    )
    net.connect(n1.id, n2.id, weight=1.0, delay=1)

    for t in range(1, 11):
        net.enqueue([Event(token_id=1, t=t, src_id=n1.id, meta={"weight": 40.0})])
        net.tick(t, dt=1.0)

    assert n3.alive, "孤立节点应保持存活"
    assert n3.energy == 3.0, "孤立节点无输入无衰减，能量应保持不变"
    assert n3.spike_count == 0, "孤立节点不应放电"


def test_negative_weight_clamped_to_zero() -> None:
    """负权重连接应被钳制为 0，且不产生传播。"""
    net, a, b = _pair()
    net.connect(a.id, b.id, weight=-5.0, delay=1)
    assert net.neurons[a.id].links[b.id]["weight"] == 0.0

    net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 40.0})])
    net.tick(1)
    assert net.total_propagations == 0, "零权重边不应产生传播事件"
    assert b.spike_count == 0, "目标不应收到电流"


def test_zero_weight_edge_does_not_propagate() -> None:
    """显式零权重边不应产生传播。"""
    net, a, b = _pair()
    net.connect(a.id, b.id, weight=0.0, delay=1)

    net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 40.0})])
    net.tick(1)
    assert net.total_propagations == 0
    assert b.spike_count == 0


def test_duplicate_connection_overwrites_not_accumulates() -> None:
    """重复连接同一对神经元应幂等覆盖，而非累积多条边。"""
    net, a, b = _pair()
    net.connect(a.id, b.id, weight=1.0, delay=1)
    net.connect(a.id, b.id, weight=0.5, delay=2)

    links = net.neurons[a.id].links
    assert len(links) == 1, "重复连接不应产生重复边"
    assert links[b.id] == {"weight": 0.5, "delay": 2}


def test_unknown_source_event_ignored() -> None:
    """src_id 不在网络中的事件（已移除神经元的残留事件）应被忽略而非崩溃。"""
    net, _, _ = _pair()
    net.enqueue([Event(token_id=1, t=1, src_id="ghost", meta={"weight": 40.0})])

    snapshot = net.tick(1)
    assert snapshot["ready_events"] == 1, "事件应从队列取出"
    assert snapshot["tick_spikes"] == 0, "未知源事件不应触发放电"
    assert net.total_spikes == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weight_event_sanitized(bad: float) -> None:
    """NaN/Inf 权重事件：safe_float 兜底为 0，网络状态不被污染。"""
    net, a, b = _pair()
    net.connect(a.id, b.id, weight=1.0, delay=1)

    net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": bad})])
    snapshot = net.tick(1)

    assert net.total_spikes == 0, "非有限权重不应触发放电"
    assert math.isfinite(snapshot["mean_energy"]), "快照能量不应为 NaN/Inf"
    for neuron in net.neurons.values():
        assert math.isfinite(neuron.energy), f"{neuron.id} 能量被污染"
        assert math.isfinite(neuron.v), f"{neuron.id} 电位被污染"
