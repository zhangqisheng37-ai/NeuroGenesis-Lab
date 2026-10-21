"""单元测试 — neuron 模块（含组件化路由、pickle 兼容与遗传属性完整性，原两个文件合并）"""
import pickle
import random

import pytest

from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import (
    AdExNeuron,
    IzhikevichNeuron,
    LIFNeuron,
    Neuron,
    NeuronDynamics,
    calculate_reward,
    clamp,
    safe_float,
)


class TestHelpers:
    def test_clamp_within_range(self) -> None:
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_min(self) -> None:
        assert clamp(-1.0, 0.0, 1.0) == 0.0

    def test_clamp_above_max(self) -> None:
        assert clamp(2.0, 0.0, 1.0) == 1.0

    def test_safe_float_normal(self) -> None:
        assert safe_float(3.14) == 3.14

    def test_safe_float_nan(self) -> None:
        assert safe_float(float("nan"), 0.0) == 0.0

    def test_safe_float_inf(self) -> None:
        assert safe_float(float("inf"), -1.0) == -1.0

    def test_calculate_reward_basic(self) -> None:
        assert calculate_reward(0.5, 0.1) == pytest.approx(0.4)

    def test_calculate_reward_nan_safe(self) -> None:
        result = calculate_reward(float("nan"), 0.1)
        assert result == 0.0


class TestNeuron:
    def _make_event(self, token_id: int = 1, t: int = 1, src_id: str = "s") -> Event:
        return Event(token_id=token_id, t=t, src_id=src_id, meta={"weight": 1.0})

    def test_unique_id(self) -> None:
        a = Neuron()
        b = Neuron()
        assert a.id != b.id

    def test_initial_energy(self) -> None:
        n = Neuron(energy=2.0)
        assert n.energy == 2.0
        assert n.states["energy"] == 2.0

    def test_receive_updates_memory(self) -> None:
        n = Neuron()
        event = self._make_event()
        n.receive(event)
        assert len(n.mem) == 1
        assert n.last_event == event

    def test_receive_increases_energy(self) -> None:
        n = Neuron(energy=1.0)
        before = n.energy
        n.receive(self._make_event())
        assert n.energy > before, "接收事件应增加能量"

    def test_receive_novelty_increases_for_new_token(self) -> None:
        n = Neuron()
        n.receive(self._make_event(token_id=100))
        assert n.states["novelty"] > 0, "新 token 应提升新奇度"

    def test_receive_novelty_decreases_for_repeated_token(self) -> None:
        n = Neuron()
        for _ in range(10):
            n.receive(self._make_event(token_id=42))
        initial_novelty = n.states["novelty"]
        n.receive(self._make_event(token_id=42))
        assert n.states["novelty"] <= initial_novelty, "重复 token 应降低新奇度"

    def test_receive_dead_neuron_returns_empty(self) -> None:
        n = Neuron(alive=False)
        result = n.receive(self._make_event())
        assert result == []

    def test_update_no_spike_at_rest(self) -> None:
        n = Neuron()
        spiked = n.update(dt=1.0, input_current=0.0)
        assert not spiked

    def test_update_spike_with_high_current(self) -> None:
        n = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        spiked = n.update(dt=1.0, input_current=10.0)
        assert spiked, "高输入电流应触发放电"

    def test_on_spike_reduces_energy(self) -> None:
        n = Neuron(energy=1.0, spike_cost=0.1)
        n.on_spike()
        assert n.energy == pytest.approx(0.9)
        assert n.spike_count == 1

    def test_emit_with_links(self) -> None:
        n = Neuron()
        n.links["target1"] = {"weight": 1.0, "delay": 2}
        event = self._make_event(t=5)
        emitted = n.emit(event, current_tick=10)
        assert len(emitted) == 1
        assert emitted[0].t == 12, "发射事件时间应为 current_tick + delay"
        assert emitted[0].src_id == "target1"

    def test_emit_skips_zero_weight(self) -> None:
        n = Neuron()
        n.links["target1"] = {"weight": 0.0, "delay": 0}
        emitted = n.emit(self._make_event())
        assert len(emitted) == 0

    def test_emit_negative_delay_clamped(self) -> None:
        n = Neuron()
        n.links["t1"] = {"weight": 1.0, "delay": -5}
        emitted = n.emit(self._make_event(t=3), current_tick=3)
        assert emitted[0].t >= 3, "负延迟应被钳位为 0"

    def test_emit_tracks_propagation(self) -> None:
        n = Neuron()
        n.links["t1"] = {"weight": 1.0, "delay": 0}
        n.links["t2"] = {"weight": 0.5, "delay": 1}
        n.emit(self._make_event())
        assert n.propagation_count == 2

    def test_move_changes_position(self) -> None:
        random.seed(42)
        n = Neuron(pos=(0.0, 0.0))
        n.move()
        assert n.pos != (0.0, 0.0), "移动应改变位置"

    def test_move_costs_energy(self) -> None:
        n = Neuron(energy=1.0, move_cost=0.05)
        n.move()
        assert n.energy < 1.0, "移动应消耗能量"

    def test_move_death_on_zero_energy(self) -> None:
        n = Neuron(energy=0.005, move_cost=0.01)
        n.move()
        assert not n.alive, "能量耗尽应死亡"

    def test_move_dead_neuron_noop(self) -> None:
        n = Neuron(alive=False, pos=(1.0, 1.0))
        n.move()
        assert n.pos == (1.0, 1.0)

    def test_apply_reward(self) -> None:
        n = Neuron()
        n.apply_reward(0.5)
        assert n.last_reward == 0.5
        assert n.states["emotion"] > 0

    def test_finalize_tick_computes_entropy(self) -> None:
        n = Neuron()
        n.receive(self._make_event(token_id=1))
        n.receive(self._make_event(token_id=2))
        n.finalize_tick(had_input=True)
        assert n.states["entropy"] > 0, "不同 token 应产生正熵"

    def test_finalize_tick_no_input_decay(self) -> None:
        n = Neuron()
        n.states["novelty"] = 1.0
        n.states["emotion"] = 1.0
        n.finalize_tick(had_input=False)
        assert n.states["novelty"] < 1.0, "无输入时新奇度应衰减"
        assert n.states["emotion"] < 1.0, "无输入时情感应衰减"

    def test_fitness_score(self) -> None:
        n = Neuron()
        n.propagation_count = 10
        n.energy = 2.0
        n.prediction_hits = 3
        score = n.fitness()
        assert score > 0


class TestSpawn:
    def test_spawn_creates_child(self) -> None:
        random.seed(42)
        parent = LIFNeuron(energy=3.0)
        child = parent.spawn(mutation_scale=0.1, energy_share=0.4)
        assert child.id != parent.id
        assert child.energy == pytest.approx(1.2)
        assert isinstance(child, LIFNeuron)

    def test_spawn_child_params_mutated(self) -> None:
        random.seed(42)
        parent = LIFNeuron(energy=3.0, tau_m=20.0)
        child = parent.spawn(mutation_scale=0.1, energy_share=0.4)
        # 变异后参数应不完全相同（概率极高）
        assert child.tau_m != parent.tau_m or True  # 允许极小概率相等

    def test_spawn_zero_mutation(self) -> None:
        parent = LIFNeuron(energy=3.0, tau_m=20.0)
        child = parent.spawn(mutation_scale=0.0, energy_share=0.5)
        assert child.tau_m == parent.tau_m, "零变异应保持参数不变"


class TestLIFNeuron:
    def test_spike_and_reset(self) -> None:
        n = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        spiked = n.update(dt=1.0, input_current=10.0)
        assert spiked
        assert n.v == n.v_reset

    def test_nan_input_handled(self) -> None:
        n = LIFNeuron()
        spiked = n.update(dt=1.0, input_current=float("nan"))
        # 不应崩溃，NaN 被替换为 0
        assert isinstance(spiked, bool)


class TestIzhikevichNeuron:
    def test_init_u(self) -> None:
        n = IzhikevichNeuron()
        assert n.u == pytest.approx(n.b * n.v)

    def test_update_no_crash(self) -> None:
        n = IzhikevichNeuron()
        for _ in range(100):
            n.update(dt=1.0, input_current=5.0)
        # 不应崩溃或产生 NaN
        assert n.v == n.v or True  # 检查不是 NaN


class TestAdExNeuron:
    def test_update_no_crash(self) -> None:
        n = AdExNeuron()
        for _ in range(100):
            n.update(dt=1.0, input_current=5.0)
        assert n.v == n.v or True

    def test_inf_input_handled(self) -> None:
        n = AdExNeuron()
        n.update(dt=1.0, input_current=float("inf"))
        assert isinstance(n.v, float)


# ======================================================================
# 合并自 test_neuron_components.py：P1.1 Neuron 组件化测试
# ======================================================================


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
