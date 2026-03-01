"""VectorizedNetwork 与 AutoLearner 集成测试。"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from neurogenesis_lab.learning import AutoLearner, RuleDiscovery, RuleStore
from neurogenesis_lab.vectorized import VectorizedNetwork


def _make_vectorized_net(n: int = 6) -> VectorizedNetwork:
    net = VectorizedNetwork(energy_decay=0.0, move_cost=0.0, min_neurons=0)
    net.init_neurons(n, energy_init=1.0)
    return net


def _write_seq(net: VectorizedNetwork, idx: int, seq: list[int]) -> None:
    for token in seq:
        net._write_token(idx, token)


def test_vectorized_snapshot_can_feed_autolearner() -> None:
    """VectorizedNetwork.tick() 的快照可直接交给 AutoLearner.on_tick()。"""
    np.random.seed(42)
    net = _make_vectorized_net(6)

    # 准备可学习样本（满足 min_samples=3）
    _write_seq(net, 0, [49, 43, 50, 61, 51])  # 1+2=3
    _write_seq(net, 1, [50, 43, 51, 61, 53])  # 2+3=5
    _write_seq(net, 2, [51, 43, 52, 61, 55])  # 3+4=7

    auto = AutoLearner(discovery=RuleDiscovery(min_support=1), learn_interval=1)
    snapshot = net.tick(1)
    report = auto.on_tick(1, snapshot, net)

    assert "rule_count" in report
    assert "prediction_accuracy" in report
    assert "tick_predictions" in report
    assert "tick_correct" in report

    # 模拟 CLI 注入行为
    snapshot["rule_count"] = report["rule_count"]
    snapshot["prediction_accuracy"] = report["prediction_accuracy"]
    assert isinstance(snapshot["rule_count"], int)
    assert isinstance(snapshot["prediction_accuracy"], float)


def test_vectorized_prediction_reward_and_hits() -> None:
    """命中预测应增加能量并累计 prediction_hits。"""
    net = _make_vectorized_net(2)
    net.energy[:] = 1.0

    # 让规则组合器可根据 bigram 预测最后一个 token
    auto = AutoLearner(
        discovery=RuleDiscovery(min_support=1),
        prediction_reward=0.5,
        prediction_penalty=0.2,
    )
    auto.store = RuleStore(
        trigram_support=OrderedDict(),
        bigram_support=OrderedDict({(3, 4): 10}),
    )
    auto.rule_count = 1

    _write_seq(net, 0, [1, 2, 3, 4])  # 预测命中
    _write_seq(net, 1, [1, 2, 3, 9])  # 预测错误

    snapshot = net.tick(1)
    report = auto.on_tick(1, snapshot, net)

    assert report["tick_predictions"] == 2
    assert report["tick_correct"] >= 1
    assert net.energy[0] > 1.0
    assert net.prediction_hits[0] >= 1


def test_vectorized_penalty_never_below_zero() -> None:
    """预测错误惩罚后能量不应低于 0。"""
    net = _make_vectorized_net(1)
    net.energy[0] = 0.01

    auto = AutoLearner(
        discovery=RuleDiscovery(min_support=1),
        prediction_reward=0.0,
        prediction_penalty=1.0,
    )
    auto.store = RuleStore(
        trigram_support=OrderedDict(),
        bigram_support=OrderedDict({(7, 8): 5}),
    )
    auto.rule_count = 1

    _write_seq(net, 0, [1, 2, 7, 9])  # 预测应为 8，实际为 9

    snapshot = net.tick(1)
    auto.on_tick(1, snapshot, net)

    assert float(net.energy[0]) >= 0.0
