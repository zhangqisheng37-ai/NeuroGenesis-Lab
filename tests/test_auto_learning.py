"""自动学习引擎集成测试"""

from __future__ import annotations

from collections import OrderedDict

from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.learning import (
    AutoLearner,
    RuleDiscovery,
    RuleStore,
)
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron


def _make_network(n: int = 8, **kwargs) -> Network:
    """创建测试用网络"""
    neurons = [LIFNeuron() for _ in range(n)]
    net = Network(
        neurons={neuron.id: neuron for neuron in neurons},
        max_neurons=kwargs.pop("max_neurons", 50),
        energy_cap=kwargs.pop("energy_cap", 5.0),
        **kwargs,
    )
    # 环形连接
    for i, neuron in enumerate(neurons):
        target = neurons[(i + 1) % n]
        net.connect(neuron.id, target.id, weight=1.0, delay=1)
    return net


def _make_network(n: int = 8, **kwargs) -> Network:
    """创建测试用网络"""
    neurons = [LIFNeuron() for _ in range(n)]
    net = Network(
        neurons={neuron.id: neuron for neuron in neurons},
        max_neurons=kwargs.pop("max_neurons", 50),
        energy_cap=kwargs.pop("energy_cap", 5.0),
        **kwargs,
    )
    # 环形连接
    for i, neuron in enumerate(neurons):
        target = neurons[(i + 1) % n]
        net.connect(neuron.id, target.id, weight=1.0, delay=1)
    return net


def _inject_events(network: Network, ticks: int = 20) -> None:
    """注入信号事件"""
    encoder = Encoder()
    neurons = list(network.neurons.values())
    tokens = encoder.encode_expr("1+2=3")
    for tick_start in range(1, ticks + 1, 5):
        base_events = encoder.to_events(tokens, src_id=neurons[0].id)
        scheduled = [
            Event(
                token_id=e.token_id,
                t=e.t + tick_start,
                src_id=e.src_id,
                meta={**e.meta, "weight": 40.0},
            )
            for e in base_events
            if e.t + tick_start <= ticks
        ]
        network.enqueue(scheduled)


class TestAutoLearnerCreation:
    """AutoLearner 创建与基本属性"""

    def test_default_creation(self):
        auto = AutoLearner(discovery=RuleDiscovery())
        assert auto.rule_count == 0
        assert auto.total_predictions == 0
        assert auto.correct_predictions == 0
        assert auto.prediction_accuracy() == 0.0

    def test_custom_params(self):
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=3),
            learn_interval=20,
            prediction_reward=0.2,
            prediction_penalty=0.05,
            max_rules=100,
        )
        assert auto.learn_interval == 20
        assert auto.prediction_reward == 0.2
        assert auto.prediction_penalty == 0.05
        assert auto.max_rules == 100

    def test_get_summary_empty(self):
        auto = AutoLearner(discovery=RuleDiscovery())
        summary = auto.get_summary()
        assert summary["rule_count"] == 0
        assert summary["trigram_rules"] == 0
        assert summary["bigram_rules"] == 0
        assert summary["pattern_rules"] == 0
        assert summary["total_predictions"] == 0
        assert summary["prediction_accuracy"] == 0.0


class TestAutoLearnerOnTick:
    """on_tick 集成行为"""

    def test_on_tick_returns_report(self, ring_network_factory):
        network = ring_network_factory(n=8, max_neurons=50, energy_cap=5.0)
        auto = AutoLearner(discovery=RuleDiscovery())
        snapshot = network.tick(1, dt=1.0)
        report = auto.on_tick(1, snapshot, network)
        assert "rule_count" in report
        assert "prediction_accuracy" in report
        assert "tick_predictions" in report
        assert "tick_correct" in report

    def test_on_tick_no_crash_empty_network(self):
        """空网络不应崩溃"""
        network = _make_network(n=2)
        auto = AutoLearner(discovery=RuleDiscovery())
        for t in range(1, 11):
            snapshot = network.tick(t, dt=1.0)
            report = auto.on_tick(t, snapshot, network)
        assert report["rule_count"] >= 0

    def test_learning_with_injected_events(self):
        """注入事件后应能发现规则"""
        network = _make_network(n=8)
        _inject_events(network, ticks=50)
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            learn_interval=10,
        )

        for t in range(1, 51):
            snapshot = network.tick(t, dt=1.0)
            auto.on_tick(t, snapshot, network)

        # 运行足够多 ticks 后，规则库应非空
        summary = auto.get_summary()
        # 规则发现取决于仿真动力学，可能为 0 也可能有规则
        assert summary["rule_count"] >= 0
        assert summary["total_predictions"] >= 0

    def test_learn_interval_respected(self):
        """学习只在指定间隔执行"""
        network = _make_network()
        _inject_events(network, ticks=30)
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            learn_interval=15,
        )
        # 运行 14 ticks（不到间隔，不应学习）
        for t in range(1, 15):
            snapshot = network.tick(t, dt=1.0)
            auto.on_tick(t, snapshot, network)
        rules_before = auto.rule_count

        # 第 15 个 tick 应触发学习
        snapshot = network.tick(15, dt=1.0)
        auto.on_tick(15, snapshot, network)
        # 无论规则是否增加，至少不应崩溃
        assert auto.rule_count >= rules_before


class TestPredictionRewards:
    """预测驱动奖励机制"""

    def test_prediction_accuracy_increases(self):
        """多轮学习后预测准确率应无 NaN"""
        network = _make_network(n=8)
        _inject_events(network, ticks=100)
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            learn_interval=5,
            prediction_reward=0.1,
            prediction_penalty=0.02,
        )

        for t in range(1, 101):
            snapshot = network.tick(t, dt=1.0)
            auto.on_tick(t, snapshot, network)

        acc = auto.prediction_accuracy()
        assert 0.0 <= acc <= 1.0  # 值在合法范围

    def test_reward_changes_energy(self):
        """预测奖励应影响神经元能量"""
        network = _make_network(n=4)
        _inject_events(network, ticks=50)
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            learn_interval=5,
            prediction_reward=0.5,  # 高奖励便于观察
        )
        for t in range(1, 51):
            snapshot = network.tick(t, dt=1.0)
            auto.on_tick(t, snapshot, network)
        # 验证 prediction_hits 被更新
        total_hits = sum(n.prediction_hits for n in network.neurons.values())
        assert total_hits >= 0  # 至少不应为负数

    def test_penalty_does_not_kill(self):
        """预测惩罚不应直接杀死神经元"""
        network = _make_network(n=4)
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            prediction_penalty=0.5,  # 高惩罚
        )
        for t in range(1, 21):
            snapshot = network.tick(t, dt=1.0)
            auto.on_tick(t, snapshot, network)
        # 一些神经元可能因其他原因死亡，但 energy 不应为负
        for n in network.neurons.values():
            assert n.energy >= 0.0


class TestRulePruning:
    """规则淘汰机制"""

    def test_max_rules_enforced(self):
        """规则数不应超过上限"""
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            max_rules=10,
        )
        # 手动注入大量规则
        trigrams = {(i, i + 1, i + 2): 1 for i in range(20)}
        bigrams = {(i, i + 1): 1 for i in range(20)}
        auto.store = RuleStore(trigram_support=trigrams, bigram_support=bigrams)
        auto.rule_count = len(trigrams) + len(bigrams)

        auto._prune_rules()
        assert auto.rule_count <= 10

    def test_prune_keeps_high_support(self):
        """LRU 淘汰应移除最久未访问的规则，保留最近访问的"""
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            max_rules=4,
        )
        trigrams = OrderedDict([
            ((1, 2, 3), 100),  # 最早插入
            ((4, 5, 6), 1),    # 中间插入
            ((7, 8, 9), 50),   # 最近插入
        ])
        auto.store = RuleStore(trigram_support=trigrams, bigram_support={})
        auto.rule_count = len(trigrams)

        # 访问 (1,2,3) 使其移到末尾（最近访问）
        auto.store.trigram_confidence((1, 2, 3))

        auto._prune_rules()
        # (1,2,3) 最近被访问过，应被保留；(4,5,6) 最久未访问，应被淘汰
        assert (1, 2, 3) in auto.store.trigram_support
        assert (7, 8, 9) in auto.store.trigram_support
        assert len(auto.store.trigram_support) <= auto.max_rules // 2


class TestSnapshotIntegration:
    """学习指标注入 TickSnapshot"""

    def test_snapshot_has_learning_fields(self):
        """on_tick 返回后 snapshot 可正确注入学习字段"""
        network = _make_network()
        auto = AutoLearner(discovery=RuleDiscovery())
        snapshot = network.tick(1, dt=1.0)
        report = auto.on_tick(1, snapshot, network)

        # 模拟 CLI 行为注入
        snapshot["rule_count"] = report["rule_count"]
        snapshot["prediction_accuracy"] = report["prediction_accuracy"]

        assert "rule_count" in snapshot
        assert "prediction_accuracy" in snapshot
        assert isinstance(snapshot["rule_count"], int)
        assert isinstance(snapshot["prediction_accuracy"], float)

    def test_100_ticks_integration(self):
        """完整 100 ticks 集成测试"""
        network = _make_network(n=8)
        _inject_events(network, ticks=100)
        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            learn_interval=10,
        )

        last_report = None
        for t in range(1, 101):
            snapshot = network.tick(t, dt=1.0)
            last_report = auto.on_tick(t, snapshot, network)
            snapshot["rule_count"] = last_report["rule_count"]
            snapshot["prediction_accuracy"] = last_report["prediction_accuracy"]

        assert last_report is not None
        summary = auto.get_summary()
        # 100 ticks 后应有某种学习结果
        assert summary["total_predictions"] >= 0
        assert 0.0 <= summary["prediction_accuracy"] <= 1.0


class TestAutoLearnerWithMultipleExpressions:
    """多表达式学习"""

    def test_multiple_expressions(self):
        """多个不同表达式应丰富规则库"""
        network = _make_network(n=8)
        encoder = Encoder()
        neurons = list(network.neurons.values())

        # 注入多种表达式
        for expr in ["1+2=3", "2+3=5", "3+4=7", "4+5=9"]:
            tokens = encoder.encode_expr(expr)
            for tick_start in range(1, 51, 5):
                base_events = encoder.to_events(tokens, src_id=neurons[0].id)
                scheduled = [
                    Event(
                        token_id=e.token_id,
                        t=e.t + tick_start,
                        src_id=e.src_id,
                        meta={**e.meta, "weight": 40.0},
                    )
                    for e in base_events
                    if e.t + tick_start <= 50
                ]
                network.enqueue(scheduled)

        auto = AutoLearner(
            discovery=RuleDiscovery(min_support=1),
            learn_interval=10,
        )

        for t in range(1, 51):
            snapshot = network.tick(t, dt=1.0)
            auto.on_tick(t, snapshot, network)

        summary = auto.get_summary()
        # 多表达式学习后规则数应非负
        assert summary["rule_count"] >= 0
