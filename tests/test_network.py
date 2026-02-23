"""单元测试 — network 模块"""

import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import pytest
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.network import Corpse, EventQueue, Network
from neurogenesis_lab.neuron import LIFNeuron, IzhikevichNeuron, AdExNeuron


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
