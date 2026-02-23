"""P1.1 Neuron 组件化测试"""

import pickle

from neurogenesis_lab.config import HERITABLE_ATTRS
from neurogenesis_lab.neuron import (
    AdExNeuron,
    IzhikevichNeuron,
    LIFNeuron,
    Neuron,
    NeuronDynamics,
    NeuronEvolution,
    NeuronMemory,
)


class TestComponentAccessibility:
    """组件可访问性 — n._dynamics.tau_m == n.tau_m"""

    def test_dynamics_fields_accessible(self) -> None:
        n = Neuron(tau_m=15.0, r_m=2.0, v_th=-50.0)
        assert n._dynamics.tau_m == n.tau_m == 15.0
        assert n._dynamics.r_m == n.r_m == 2.0
        assert n._dynamics.v_th == n.v_th == -50.0

    def test_evolution_fields_accessible(self) -> None:
        n = Neuron(spike_count=5, propagation_count=10, prediction_hits=3)
        assert n._evolution.spike_count == n.spike_count == 5
        assert n._evolution.propagation_count == n.propagation_count == 10
        assert n._evolution.prediction_hits == n.prediction_hits == 3

    def test_memory_fields_accessible(self) -> None:
        n = Neuron(mem_size=32)
        assert n._memory.mem_size == n.mem_size == 32
        assert n.mem.maxlen == 32


class TestSetAttrRouting:
    """setattr 路由正确性"""

    def test_set_dynamics_field(self) -> None:
        n = Neuron()
        n.tau_m = 30.0
        assert n._dynamics.tau_m == 30.0
        assert n.tau_m == 30.0

    def test_set_evolution_field(self) -> None:
        n = Neuron()
        n.spike_count = 42
        assert n._evolution.spike_count == 42
        assert n.spike_count == 42

    def test_set_memory_field(self) -> None:
        n = Neuron()
        n._observe_dirty = False
        assert n._memory._observe_dirty is False

    def test_set_direct_field(self) -> None:
        n = Neuron()
        n.energy = 0.5
        assert n.energy == 0.5
        # energy 不在组件中，应在 __dict__ 中
        assert "energy" in n.__dict__


class TestSpawnComponentIndependence:
    """spawn 组件独立性"""

    def test_child_has_independent_dynamics(self) -> None:
        parent = Neuron(tau_m=20.0, energy=3.0)
        child = parent.spawn(0.0, 0.5)  # mutation_scale=0 保证精确复制
        # 修改子代不影响父代
        child.tau_m = 999.0
        assert parent.tau_m == 20.0

    def test_child_has_independent_evolution(self) -> None:
        parent = Neuron(energy=3.0)
        child = parent.spawn(0.0, 0.5)
        child.spike_count = 100
        assert parent.spike_count == 0


class TestPickleRoundTrip:
    """pickle 新旧格式往返"""

    def test_pickle_roundtrip(self) -> None:
        n = Neuron(tau_m=15.0, spike_count=7, mem_size=32)
        n.energy = 0.8
        data = pickle.dumps(n)
        restored = pickle.loads(data)
        assert restored.tau_m == 15.0
        assert restored.spike_count == 7
        assert restored.mem_size == 32
        assert restored.energy == 0.8
        assert hasattr(restored, "_dynamics")
        assert hasattr(restored, "_evolution")
        assert hasattr(restored, "_memory")

    def test_old_format_pickle_compat(self) -> None:
        """模拟旧格式 pickle（无组件字段）"""
        n = Neuron(tau_m=15.0, spike_count=7)
        # 手动构造旧格式 state
        old_state = dict(n.__dict__)
        # 移除组件，添加扁平字段
        old_state.pop("_dynamics", None)
        old_state.pop("_evolution", None)
        old_state.pop("_memory", None)
        old_state["tau_m"] = 15.0
        old_state["r_m"] = 1.0
        old_state["v"] = -65.0
        old_state["u"] = 0.0
        old_state["w"] = 0.0
        old_state["v_rest"] = -65.0
        old_state["v_reset"] = -65.0
        old_state["v_th"] = -55.0
        old_state["spike_cost"] = 0.05
        old_state["input_gain"] = 0.05
        old_state["move_cost"] = 0.01
        old_state["spike_count"] = 7
        old_state["propagation_count"] = 0
        old_state["prediction_hits"] = 0
        old_state["mem_size"] = 64
        from collections import Counter, deque
        old_state["mem"] = deque(maxlen=64)
        old_state["transitions"] = Counter()
        old_state["_observe_cache"] = []
        old_state["_observe_dirty"] = True

        restored = Neuron.__new__(Neuron)
        restored.__setstate__(old_state)
        assert restored.tau_m == 15.0
        assert restored.spike_count == 7
        assert hasattr(restored, "_dynamics")


class TestSubclassCompatibility:
    """子类额外字段不在 dynamics 中"""

    def test_izhikevich_extra_fields_not_in_dynamics(self) -> None:
        n = IzhikevichNeuron(a=0.03, b=0.25)
        # a, b 是子类字段，应在 __dict__ 中而非 _dynamics
        assert n.a == 0.03
        assert n.b == 0.25
        assert not hasattr(n._dynamics, "a")
        assert not hasattr(n._dynamics, "b")

    def test_adex_extra_fields_not_in_dynamics(self) -> None:
        n = AdExNeuron(delta_t=3.0, tau_w=100.0)
        assert n.delta_t == 3.0
        assert n.tau_w == 100.0
        assert not hasattr(n._dynamics, "delta_t")
        assert not hasattr(n._dynamics, "tau_w")

    def test_lif_neuron_has_components(self) -> None:
        n = LIFNeuron(tau_m=25.0)
        assert n._dynamics.tau_m == 25.0
        assert hasattr(n, "_evolution")
        assert hasattr(n, "_memory")


class TestHeritableAttrsCompleteness:
    """HERITABLE_ATTRS 常量完整性"""

    def test_all_dynamics_attrs_heritable(self) -> None:
        """NeuronDynamics 中除 v/u/w 外的字段都应在 HERITABLE_DYNAMICS_ATTRS 中"""
        from neurogenesis_lab.config import HERITABLE_DYNAMICS_ATTRS
        dynamics_fields = set(NeuronDynamics.__dataclass_fields__.keys())
        # v, u, w 是状态变量，不遗传
        non_heritable = {"v", "u", "w"}
        heritable_dynamics = dynamics_fields - non_heritable
        assert heritable_dynamics == set(HERITABLE_DYNAMICS_ATTRS)

    def test_subclass_attrs_match_izhikevich_and_adex(self) -> None:
        """HERITABLE_SUBCLASS_ATTRS 应覆盖 Izhikevich 和 AdEx 的额外字段"""
        from neurogenesis_lab.config import HERITABLE_SUBCLASS_ATTRS
        izh_fields = set(IzhikevichNeuron.__dataclass_fields__.keys()) - set(
            Neuron.__dataclass_fields__.keys()
        )
        adex_fields = set(AdExNeuron.__dataclass_fields__.keys()) - set(
            Neuron.__dataclass_fields__.keys()
        )
        all_subclass = izh_fields | adex_fields
        assert all_subclass == set(HERITABLE_SUBCLASS_ATTRS)
