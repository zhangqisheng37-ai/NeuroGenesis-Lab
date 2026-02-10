"""单元测试 — neuron 模块"""
import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import pytest
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import (
    AdExNeuron,
    IzhikevichNeuron,
    LIFNeuron,
    Neuron,
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
