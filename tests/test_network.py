"""单元测试 — network 模块（含 config/state 分离、空间连接与拓扑真实边界，原四个文件合并）"""

import math
import pickle
import random
import tempfile
from pathlib import Path

import pytest

from neurogenesis_lab.encoding import Event
from neurogenesis_lab.network import Corpse, EventQueue, Network, NetworkConfig, NetworkState
from neurogenesis_lab.neuron import LIFNeuron


class TestEventQueue:
    def test_push_and_pop(self) -> None:
        q = EventQueue()
        e1 = Event(token_id=1, t=3, src_id="s", meta={})
        e2 = Event(token_id=2, t=1, src_id="s", meta={})
        q.push(e1)
        q.push(e2)
        ready = q.pop_ready(2)
        assert len(ready) == 1
        assert ready[0].token_id == 2

    def test_pop_all(self) -> None:
        q = EventQueue()
        for i in range(5):
            q.push(Event(token_id=i, t=i, src_id="s", meta={}))
        ready = q.pop_ready(10)
        assert len(ready) == 5

    def test_push_many(self) -> None:
        q = EventQueue()
        events = [Event(token_id=i, t=i, src_id="s", meta={}) for i in range(3)]
        q.push_many(events)
        assert len(q) == 3


class TestNetwork:
    def _make_network(self, n: int = 4, **kwargs) -> tuple[Network, list[LIFNeuron]]:
        neurons = [LIFNeuron() for _ in range(n)]
        net = Network(
            neurons={n.id: n for n in neurons},
            **kwargs,
        )
        return net, neurons

    def test_connect(self) -> None:
        net, neurons = self._make_network(2)
        net.connect(neurons[0].id, neurons[1].id, weight=0.5, delay=3)
        assert neurons[0].links[neurons[1].id]["weight"] == 0.5
        assert neurons[0].links[neurons[1].id]["delay"] == 3

    def test_connect_negative_delay_clamped(self) -> None:
        net, neurons = self._make_network(2)
        net.connect(neurons[0].id, neurons[1].id, weight=1.0, delay=-5)
        assert neurons[0].links[neurons[1].id]["delay"] == 0, "负延迟应被钳位为 0"

    def test_tick_returns_snapshot(self) -> None:
        net, _ = self._make_network(2)
        snapshot = net.tick(1)
        assert "tick" in snapshot
        assert "alive_neurons" in snapshot
        assert "schema_version" in snapshot
        assert snapshot["tick"] == 1

    def test_tick_snapshot_has_births_deaths(self) -> None:
        net, _ = self._make_network(2)
        snapshot = net.tick(1)
        assert "births" in snapshot
        assert "deaths" in snapshot

    def test_event_delivery(self) -> None:
        """事件应按时间正确投递"""
        net, neurons = self._make_network(2)
        event = Event(token_id=1, t=3, src_id=neurons[0].id, meta={"weight": 1.0})
        net.enqueue([event])

        s1 = net.tick(1)
        assert s1["ready_events"] == 0, "tick 1 不应有就绪事件"

        s3 = net.tick(3)
        assert s3["ready_events"] >= 1, "tick 3 应投递事件"


class TestSpikeAndPropagation:
    def test_spike_triggers_emission(self) -> None:
        """放电应触发向邻居传播"""
        random.seed(42)
        a = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        b = LIFNeuron()
        net = Network(
            neurons={a.id: a, b.id: b},
            input_current_scale=5.0,
        )
        net.connect(a.id, b.id, weight=1.0, delay=2)
        net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 1.0})])

        s1 = net.tick(1)
        assert s1["propagation_count"] > 0, "放电应产生传播事件"
        assert s1["tick_spikes"] > 0, "应记录 spike 计数"

    def test_propagation_with_delay(self) -> None:
        """传播事件应在 delay 后到达"""
        random.seed(42)
        a = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        b = LIFNeuron()
        net = Network(
            neurons={a.id: a, b.id: b},
            input_current_scale=5.0,
        )
        net.connect(a.id, b.id, weight=1.0, delay=2)
        net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 1.0})])

        net.tick(1)
        s2 = net.tick(2)
        assert s2["ready_events"] == 0, "delay=2 时 tick 2 不应收到传播事件"
        s3 = net.tick(3)
        assert s3["ready_events"] >= 1, "delay=2 时 tick 3 应收到传播事件"

    def test_background_current_is_consumed(self) -> None:
        """background_current 应在无外部事件时驱动膜电位上升。"""
        random.seed(42)
        neuron = LIFNeuron(tau_m=20.0, r_m=1.0, v_rest=-65.0, v_th=-50.0)
        net = Network(
            neurons={neuron.id: neuron},
            background_current=5.0,
            input_current_scale=0.0,
            energy_decay=0.0,
            move_cost=0.0,
            min_neurons=0,
            adaptive_decay=False,
        )
        v_before = neuron.v
        net.tick(1)
        assert neuron.v > v_before

    def test_spontaneous_spike_rate_is_consumed(self) -> None:
        """spontaneous_spike_rate=1 时应在无输入事件下产生放电。"""
        random.seed(42)
        neuron = LIFNeuron()
        net = Network(
            neurons={neuron.id: neuron},
            spontaneous_spike_rate=1.0,
            input_current_scale=0.0,
            energy_decay=0.0,
            move_cost=0.0,
            min_neurons=0,
            adaptive_decay=False,
        )
        snapshot = net.tick(1)
        assert snapshot["tick_spikes"] >= 1


class TestPredictionError:
    def test_pe_decreases_with_repeated_input(self) -> None:
        """PE 应随重复输入下降"""
        random.seed(42)
        a = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        b = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        net = Network(
            neurons={a.id: a, b.id: b},
            input_current_scale=5.0,
        )
        net.connect(a.id, b.id, weight=1.0, delay=1)

        # 重复注入相同事件
        pe_values = []
        for t in range(1, 21):
            net.enqueue([Event(token_id=1, t=t, src_id=a.id, meta={"weight": 1.0})])
            snapshot = net.tick(t)
            pe_values.append(snapshot["prediction_error"])

        # PE 在后期应趋于稳定或下降
        early_avg = sum(pe_values[2:5]) / 3
        late_avg = sum(pe_values[-5:]) / 5
        assert (
            late_avg <= early_avg + 0.1
        ), f"PE 应展现下降趋势: 早期平均={early_avg:.4f}, 晚期平均={late_avg:.4f}"


class TestReproductionAndDeath:
    def test_reproduction_creates_child(self) -> None:
        random.seed(42)
        parent = LIFNeuron(energy=3.5)
        net = Network(
            neurons={parent.id: parent},
            reproduction_threshold=2.5,
            reproduction_share=0.4,
            max_neurons=8,
        )
        net.tick(1)
        assert len(net.neurons) >= 2, "高能量神经元应繁殖"
        assert net.birth_count >= 1

    def test_death_creates_corpse(self) -> None:
        doomed = LIFNeuron(energy=0.0)
        net = Network(neurons={doomed.id: doomed}, max_neurons=8, min_neurons=0)
        net.tick(1)
        assert len(net.neurons) == 0
        assert len(net.corpses) >= 1
        assert net.death_count >= 1

    def test_min_neurons_prevents_extinction(self) -> None:
        """min_neurons 应阻止灭绝"""
        weak = LIFNeuron(energy=0.001)
        net = Network(neurons={weak.id: weak}, max_neurons=8, min_neurons=1)
        net.tick(1)
        assert len(net.neurons) >= 1, "min_neurons 应保护最后的神经元不被移除"

    def test_reproduction_death_balance(self) -> None:
        """长运行中网络规模应在可控区间"""
        random.seed(42)
        neurons = [LIFNeuron(energy=2.0) for _ in range(5)]
        net = Network(
            neurons={n.id: n for n in neurons},
            max_neurons=20,
            reproduction_threshold=2.0,
            min_neurons=2,
        )
        for t in range(1, 101):
            net.tick(t)
        alive = sum(1 for n in net.neurons.values() if n.alive)
        assert 1 <= alive <= 20, f"网络规模应可控，实际: {alive}"

    def test_evolution_mutation_amplitude_is_consumed(self) -> None:
        """当 evolution_mutation_amplitude 设置为 0 时应覆盖 mutation_scale。"""
        random.seed(42)
        parent = LIFNeuron(energy=3.5, tau_m=17.0, v_th=-52.0)
        net = Network(
            neurons={parent.id: parent},
            reproduction_threshold=2.0,
            reproduction_share=0.4,
            mutation_scale=0.8,
            evolution_mutation_amplitude=0.0,
            max_neurons=3,
        )
        net.tick(1)
        assert len(net.neurons) >= 2
        child = next(item for item in net.neurons.values() if item.id != parent.id)
        assert child.tau_m == parent.tau_m
        assert child.v_th == parent.v_th

    def test_energy_budget_per_tick_is_consumed(self) -> None:
        """energy_budget_per_tick 应在每个 tick 注入全局能量预算。"""
        random.seed(42)
        neuron = LIFNeuron(energy=1.0)
        net = Network(
            neurons={neuron.id: neuron},
            energy_decay=0.0,
            move_cost=0.0,
            reproduction_threshold=999.0,
            energy_budget_per_tick=0.5,
            min_neurons=0,
            max_neurons=2,
            adaptive_decay=False,
        )
        before = neuron.energy
        net.tick(1)
        assert neuron.energy > before


class TestCorpse:
    def test_corpse_energy_decay(self) -> None:
        """尸体能量应衰减"""
        corpse = Corpse(pos=(0, 0), energy=1.0, params={}, ttl=10, decay_rate=0.1)
        assert corpse.decay_rate == 0.1

    def test_scavenging(self) -> None:
        """近距离活体应吞噬尸体能量"""
        random.seed(42)
        scavenger = LIFNeuron(energy=1.0, pos=(0.0, 0.0))
        net = Network(
            neurons={scavenger.id: scavenger},
            scavenger_radius=5.0,
            min_neurons=0,
        )
        net.corpses = [Corpse(pos=(0.1, 0.1), energy=0.5, params={}, ttl=10)]
        net._update_corpses()
        assert scavenger.energy > 1.0, "近距离应吞噬尸体能量"

    def test_corpse_pool_reuse(self) -> None:
        """尸体对象在回收后应被对象池复用。"""
        random.seed(42)
        doomed = LIFNeuron(energy=0.0)
        net = Network(neurons={doomed.id: doomed}, min_neurons=0, corpse_ttl=1)

        net.tick(1)
        assert len(net.corpses) == 1

        net._update_corpses()
        assert len(net.corpses) == 0
        assert len(net.corpse_pool.free) >= 1

        another = LIFNeuron(energy=0.0)
        net.add_neuron(another)
        net.tick(2)
        assert net.corpse_pool.reused_count >= 1


class TestNetworkSummary:
    def test_get_summary(self) -> None:
        random.seed(42)
        neurons = [LIFNeuron() for _ in range(3)]
        net = Network(neurons={n.id: n for n in neurons})
        net.tick(1)
        summary = net.get_summary()
        assert "schema_version" in summary
        assert "total_neurons" in summary
        assert "alive_neurons" in summary
        assert "pe_moving_avg" in summary
        assert "birth_count" in summary
        assert "death_count" in summary


class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        """同一 seed + 同一配置应产生相同结果"""

        def run_simulation(seed: int) -> list[dict]:
            random.seed(seed)
            a = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
            b = LIFNeuron()
            net = Network(
                neurons={a.id: a, b.id: b},
                input_current_scale=5.0,
                min_neurons=0,
            )
            net.connect(a.id, b.id, weight=1.0, delay=1)
            net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 1.0})])
            results = []
            for t in range(1, 6):
                results.append(net.tick(t))
            return results

        r1 = run_simulation(42)
        r2 = run_simulation(42)
        for s1, s2 in zip(r1, r2):
            assert s1["tick"] == s2["tick"]
            assert s1["propagation_count"] == s2["propagation_count"]
            assert (
                abs(s1["prediction_error"] - s2["prediction_error"]) < 1e-10
            ), f"相同 seed 应产生相同 PE: {s1['prediction_error']} vs {s2['prediction_error']}"


class TestTickRegression:
    def test_tick_snapshot_sequence_regression(self) -> None:
        random.seed(123)
        a = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        b = LIFNeuron()
        net = Network(
            neurons={a.id: a, b.id: b},
            input_current_scale=5.0,
            energy_decay=0.0,
            move_cost=0.0,
            min_neurons=0,
            adaptive_decay=False,
        )
        net.connect(a.id, b.id, weight=1.0, delay=2)
        net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 1.0})])

        snapshots = [net.tick(t) for t in (1, 2, 3)]
        expected = [
            {
                "tick": 1,
                "ready_events": 1,
                "propagation_count": 1,
                "alive_neurons": 2,
                "tick_spikes": 1,
                "births": 0,
                "deaths": 0,
                "prediction_error": 0.0,
                "mean_energy": 0.99,
            },
            {
                "tick": 2,
                "ready_events": 0,
                "propagation_count": 0,
                "alive_neurons": 2,
                "tick_spikes": 0,
                "births": 0,
                "deaths": 0,
                "prediction_error": 0.023823517792299277,
                "mean_energy": 0.98,
            },
            {
                "tick": 3,
                "ready_events": 1,
                "propagation_count": 0,
                "alive_neurons": 2,
                "tick_spikes": 0,
                "births": 0,
                "deaths": 0,
                "prediction_error": 0.07807188484979724,
                "mean_energy": 0.995,
            },
        ]
        for snapshot, expected_tick in zip(snapshots, expected):
            for key, value in expected_tick.items():
                if isinstance(value, float):
                    assert snapshot[key] == pytest.approx(value)
                else:
                    assert snapshot[key] == value


# ======================================================================
# 合并自 test_network_config_state.py：P1.2 Network 配置/状态分离测试
# ======================================================================


class TestConfigStateAccessibility:
    """config/state 组件可访问性"""

    def test_config_fields_accessible(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100, energy_decay=0.01)
        assert net._config.max_neurons == net.max_neurons == 100
        assert net._config.energy_decay == net.energy_decay == 0.01

    def test_state_fields_accessible(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n})
        net.total_spikes = 42
        assert net._state.total_spikes == net.total_spikes == 42

    def test_all_config_fields_routed(self) -> None:
        n = LIFNeuron()
        net = Network(
            neurons={n.id: n},
            reproduction_threshold=3.0,
            reproduction_share=0.5,
            mutation_scale=0.1,
            energy_decay=0.01,
            max_neurons=100,
            max_links=300,
            corpse_ttl=30,
            scavenger_radius=3.0,
            input_current_scale=2.0,
            background_current=1.0,
            spontaneous_spike_rate=0.05,
            move_cost=0.02,
            energy_cap=10.0,
            adaptive_decay=False,
            adaptive_reproduction=False,
            endangered_threshold=5,
            congestion_penalty=0.2,
            min_neurons=3,
            energy_budget_per_tick=1.0,
        )
        assert net.reproduction_threshold == 3.0
        assert net.reproduction_share == 0.5
        assert net.mutation_scale == 0.1
        assert net.energy_decay == 0.01
        assert net.max_neurons == 100
        assert net.max_links == 300
        assert net.corpse_ttl == 30
        assert net.scavenger_radius == 3.0
        assert net.input_current_scale == 2.0
        assert net.background_current == 1.0
        assert net.spontaneous_spike_rate == 0.05
        assert net.move_cost == 0.02
        assert net.energy_cap == 10.0
        assert net.adaptive_decay is False
        assert net.adaptive_reproduction is False
        assert net.endangered_threshold == 5
        assert net.congestion_penalty == 0.2
        assert net.min_neurons == 3
        assert net.energy_budget_per_tick == 1.0


class TestSetAttrRouting:
    """setattr 路由正确性"""

    def test_set_config_field(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        net.max_neurons = 200
        assert net._config.max_neurons == 200
        assert net.max_neurons == 200

    def test_set_state_field(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        net.birth_count = 10
        assert net._state.birth_count == 10
        assert net.birth_count == 10

    def test_set_direct_field(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        net.expected_state = {"energy": 1.0}
        assert net.expected_state == {"energy": 1.0}
        assert "expected_state" in net.__dict__


class TestPickleRoundTrip:
    """pickle 新旧格式往返"""

    def test_pickle_roundtrip(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100)
        net.total_spikes = 42
        data = pickle.dumps(net)
        restored = pickle.loads(data)
        assert restored.max_neurons == 100
        assert restored.total_spikes == 42
        assert hasattr(restored, "_config")
        assert hasattr(restored, "_state")

    def test_checkpoint_roundtrip(self) -> None:
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100)
        net.tick(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ckpt"
            net.save_checkpoint(path)
            restored = Network.load_checkpoint(path)
            assert restored.max_neurons == 100
            assert hasattr(restored, "_config")
            assert hasattr(restored, "_state")

    def test_old_format_pickle_compat(self) -> None:
        """模拟旧格式 pickle（无 _config/_state）"""
        n = LIFNeuron(energy=2.0)
        net = Network(neurons={n.id: n}, max_neurons=100)
        # 手动构造旧格式 state
        old_state = dict(net.__dict__)
        config_copy = {
            k: getattr(net._config, k)
            for k in NetworkConfig.__dataclass_fields__
        }
        state_copy = {
            k: getattr(net._state, k)
            for k in NetworkState.__dataclass_fields__
        }
        old_state.pop("_config")
        old_state.pop("_state")
        old_state.update(config_copy)
        old_state.update(state_copy)

        restored = Network.__new__(Network)
        restored.__setstate__(old_state)
        assert restored.max_neurons == 100
        assert hasattr(restored, "_config")
        assert hasattr(restored, "_state")


class TestConstructionSignature:
    """构造签名不变验证"""

    def test_keyword_construction(self) -> None:
        n = LIFNeuron()
        net = Network(
            neurons={n.id: n},
            reproduction_threshold=3.0,
            max_neurons=100,
            birth_count=5,
        )
        assert net.reproduction_threshold == 3.0
        assert net.max_neurons == 100
        assert net.birth_count == 5

    def test_direct_fields_remain(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        # 这些字段应直接在 __dict__ 中
        assert "neurons" in net.__dict__
        assert "queue" in net.__dict__
        assert "replay" in net.__dict__
        assert "corpses" in net.__dict__
        assert "corpse_pool" in net.__dict__

    def test_config_state_fields_not_in_dict(self) -> None:
        n = LIFNeuron()
        net = Network(neurons={n.id: n})
        # 配置/状态字段不应在 __dict__ 中
        assert "max_neurons" not in net.__dict__
        assert "birth_count" not in net.__dict__
        assert "total_spikes" not in net.__dict__


# ======================================================================
# 合并自 test_spatial_connection.py：test_spatial_connection.py
# ======================================================================


def test_spatial_connection():
    """测试基于空间距离的连接策略"""
    network = Network(neurons={})

    # 创建 3 个神经元，位置不同
    n1 = LIFNeuron(id="n1", pos=(0.0, 0.0))
    n2 = LIFNeuron(id="n2", pos=(1.0, 0.0))  # 距离 n1 为 1.0
    n3 = LIFNeuron(id="n3", pos=(10.0, 0.0))  # 距离 n1 为 10.0

    network.add_neuron(n1)
    network.add_neuron(n2)
    network.add_neuron(n3)

    # 使用半径 5.0 进行连接
    network.connect_by_proximity(radius=5.0, max_links_per_neuron=5)

    # n1 应该连接到 n2，但不连接到 n3
    assert "n2" in n1.links
    assert "n3" not in n1.links

    # 检查权重和延迟
    link = n1.links["n2"]
    assert link["weight"] > 0.5  # 距离近，权重高
    assert link["delay"] >= 1  # 延迟至少为 1


def test_spatial_evolution():
    """测试繁殖时的空间位置继承"""
    network = Network(neurons={})
    parent = LIFNeuron(id="parent", pos=(10.0, 10.0), energy=5.0)
    network.add_neuron(parent)

    # 触发繁殖
    network.check_reproduction()

    # 应该有一个子代
    assert len(network.neurons) == 2

    # 找到子代
    child = next(n for n in network.neurons.values() if n.id != "parent")

    # 子代位置应该在母体附近（高斯扰动）
    dx = abs(child.pos[0] - parent.pos[0])
    dy = abs(child.pos[1] - parent.pos[1])

    assert dx <= 1.0
    assert dy <= 1.0


def test_neuron_movement():
    """测试神经元移动消耗能量"""
    n = LIFNeuron(id="n1", pos=(0.0, 0.0), energy=1.0, move_cost=0.1)

    # 能量充足，应该移动并消耗能量
    n.move()
    assert n.energy == 0.9
    assert n.pos != (0.0, 0.0)

    # 能量不足，不应该移动
    n.energy = 0.1
    old_pos = n.pos
    n.move()
    assert n.energy == 0.0
    assert n.pos == old_pos
    assert n.pos == old_pos


# ======================================================================
# 合并自 test_topology_realism.py：拓扑现实场景测试 — 真实世界拓扑中的非理想结构。
# ======================================================================


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
