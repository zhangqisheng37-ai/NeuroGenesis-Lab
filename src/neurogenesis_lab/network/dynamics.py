"""网络动力学步进 — network 子包的输入处理与状态聚合部分。"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from neurogenesis_lab.config import ENDANGERED_DECAY_DISCOUNT, PE_SMOOTHING_ALPHA
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.evolution import update_endangered_mode
from neurogenesis_lab.network._base import _NetworkAttrs
from neurogenesis_lab.neuron import calculate_reward, safe_float
from neurogenesis_lab.protocols import TickSnapshot


class NetworkDynamicsMixin(_NetworkAttrs):
    """事件处理、LIF 动力学更新与 tick 统计聚合。"""

    def _process_inputs(
        self, tick: int, dt: float
    ) -> tuple[list[Event], dict[str, list[Event]], float, float]:
        _ = dt
        self.current_tick = tick
        ready = self.queue.pop_ready(tick)
        inputs: dict[str, list[Event]] = defaultdict(list)
        for event in ready:
            if event.src_id not in self.neurons:
                continue
            inputs[event.src_id].append(event)

        alive_count: int = sum(1 for n in self.neurons.values() if n.alive)
        effective_decay = self._calculate_adaptive_decay(alive_count)
        budget_share = (
            self.energy_budget_per_tick / alive_count
            if self.energy_budget_per_tick > 0 and alive_count > 0
            else 0.0
        )

        self._endangered_mode = update_endangered_mode(
            self._endangered_mode, alive_count, self.endangered_threshold
        )
        return ready, inputs, effective_decay, budget_share

    def _update_dynamics(
        self,
        *,
        tick: int,
        dt: float,
        inputs: dict[str, list[Event]],
        effective_decay: float,
        budget_share: float,
    ) -> tuple[list[Event], int]:
        emitted: list[Event] = []
        tick_spikes = 0
        for neuron in list(self.neurons.values()):
            neuron.start_tick()
            events = inputs.get(neuron.id, [])
            input_current = self.background_current
            for event in events:
                neuron.receive(event)
                w = float(event.meta.get("weight", 1.0))
                input_current += safe_float(w * self.input_current_scale)

            spontaneous_spike = False
            if self.spontaneous_spike_rate > 0.0:
                spontaneous_spike = random.random() < self.spontaneous_spike_rate
            if spontaneous_spike:
                neuron.on_spike()
                spiked = True
            else:
                spiked = neuron.update(dt, input_current)

            if spiked:
                tick_spikes += 1
                if neuron.last_event is not None:
                    emitted.extend(neuron.emit(neuron.last_event, current_tick=tick))
                elif spontaneous_spike:
                    emitted.extend(
                        neuron.emit(
                            Event(
                                token_id=0,
                                t=tick,
                                src_id=neuron.id,
                                meta={"spontaneous": True, "weight": 1.0},
                            ),
                            current_tick=tick,
                        )
                    )
            neuron.move()
            decay = effective_decay * (ENDANGERED_DECAY_DISCOUNT if self._endangered_mode else 1.0)
            neuron.energy = max(0.0, neuron.energy - decay)
            if budget_share > 0.0:
                neuron.energy += budget_share
            if neuron.energy > self.energy_cap:
                neuron.energy = self.energy_cap
            energy_loss = max(0.0, neuron.last_energy - neuron.energy)
            reward = calculate_reward(neuron.states["novelty"], energy_loss)
            neuron.apply_reward(reward)
            neuron.finalize_tick(had_input=bool(events))
        return emitted, tick_spikes

    def _collect_tick_stats(
        self, *, tick: int, ready_count: int, emitted_count: int, tick_spikes: int
    ) -> TickSnapshot:
        actual_state = self._aggregate_state()
        prediction_error = self._update_prediction_error(actual_state)
        self._pe_history.append(prediction_error)

        alive_neurons = [neuron for neuron in self.neurons.values() if neuron.alive]
        alive_count = len(alive_neurons)
        mean_energy = sum(n.energy for n in alive_neurons) / alive_count if alive_count > 0 else 0.0
        snapshot: TickSnapshot = {
            "schema_version": self.schema_version,
            "tick": tick,
            "ready_events": ready_count,
            "propagation_count": emitted_count,
            "alive_neurons": alive_count,
            "total_neurons": len(self.neurons),
            "prediction_error": prediction_error,
            "tick_spikes": tick_spikes,
            "births": 0,
            "deaths": 0,
            "mean_energy": mean_energy,
        }
        self.replay.append(snapshot)
        return snapshot

    def pe_moving_average(self, window: int = 10) -> float:
        """计算预测误差 (PE) 的滑动平均值。

        Args:
            window: 滑动窗口大小（tick 数），默认 10。

        Returns:
            最近 window 个 tick 的 PE 均值，无历史时返回 0.0。
        """
        if not self._pe_history:
            return 0.0
        recent = self._pe_history[-window:]
        return sum(recent) / len(recent)

    def active_neuron_ratio(self) -> float:
        """计算当前存活神经元占总神经元的比例。

        Returns:
            存活数 / 总数，无神经元时返回 0.0。
        """
        if not self.neurons:
            return 0.0
        alive = sum(1 for n in self.neurons.values() if n.alive)
        return alive / len(self.neurons)

    def _aggregate_state(self) -> dict[str, float]:
        alive = [neuron for neuron in self.neurons.values() if neuron.alive]
        if not alive:
            return {"energy": 0.0, "entropy": 0.0, "emotion": 0.0, "novelty": 0.0}
        size = len(alive)
        return {
            "energy": sum(neuron.states["energy"] for neuron in alive) / size,
            "entropy": sum(neuron.states["entropy"] for neuron in alive) / size,
            "emotion": sum(neuron.states["emotion"] for neuron in alive) / size,
            "novelty": sum(neuron.states["novelty"] for neuron in alive) / size,
        }

    def _update_prediction_error(self, actual_state: dict[str, float]) -> float:
        if self.expected_state is None:
            self.expected_state = dict(actual_state)
            return 0.0
        prediction_error = math.sqrt(
            sum(
                (actual_state[key] - self.expected_state.get(key, 0.0)) ** 2 for key in actual_state
            )
        )
        prediction_error = safe_float(prediction_error)
        alpha = PE_SMOOTHING_ALPHA
        self.expected_state = {
            key: self.expected_state.get(key, 0.0)
            + alpha * (actual_state[key] - self.expected_state.get(key, 0.0))
            for key in actual_state
        }
        return prediction_error
