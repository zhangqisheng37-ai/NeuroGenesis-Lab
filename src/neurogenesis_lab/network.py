from __future__ import annotations

import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import Neuron, calculate_reward


@dataclass
class EventQueue:
    _heap: list[tuple[int, int, Event]] = field(default_factory=list)
    _seq: int = 0

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, (event.t, self._seq, event))
        self._seq += 1

    def push_many(self, events: Iterable[Event]) -> None:
        for event in events:
            self.push(event)

    def pop_ready(self, tick: int) -> list[Event]:
        ready: list[Event] = []
        while self._heap and self._heap[0][0] <= tick:
            _, _, event = heapq.heappop(self._heap)
            ready.append(event)
        return ready

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class Corpse:
    pos: tuple[float, float]
    energy: float
    params: dict[str, float]
    ttl: int


@dataclass
class Network:
    neurons: dict[str, Neuron]
    queue: EventQueue = field(default_factory=EventQueue)
    replay: list[dict[str, float | int]] = field(default_factory=list)
    expected_state: dict[str, float] | None = None
    corpses: list[Corpse] = field(default_factory=list)
    reproduction_threshold: float = 2.5
    reproduction_share: float = 0.4
    mutation_scale: float = 0.05
    energy_decay: float = 0.005
    max_neurons: int = 50
    max_links: int = 200
    corpse_ttl: int = 50
    scavenger_radius: float = 2.5
    input_current_scale: float = 1.0

    def connect(self, source_id: str, target_id: str, weight: float = 1.0, delay: int = 0) -> None:
        neuron = self.neurons[source_id]
        neuron.links[target_id] = {"weight": weight, "delay": delay}

    def add_neuron(self, neuron: Neuron) -> None:
        self.neurons[neuron.id] = neuron

    def enqueue(self, events: Iterable[Event]) -> None:
        self.queue.push_many(events)

    def tick(self, tick: int, dt: float = 1.0) -> dict[str, float]:
        ready = self.queue.pop_ready(tick)
        inputs: dict[str, list[Event]] = defaultdict(list)
        for event in ready:
            if event.src_id not in self.neurons:
                continue
            inputs[event.src_id].append(event)

        emitted: list[Event] = []
        for neuron in self.neurons.values():
            neuron.start_tick()
            events = inputs.get(neuron.id, [])
            input_current = 0.0
            for event in events:
                neuron.receive(event)
                input_current += float(event.meta.get("weight", 1.0)) * self.input_current_scale
            spiked = neuron.update(dt, input_current)
            if spiked and neuron.last_event is not None:
                emitted.extend(neuron.emit(neuron.last_event, current_tick=tick))
            neuron.move()
            neuron.energy = max(0.0, neuron.energy - self.energy_decay)
            energy_loss = max(0.0, neuron.last_energy - neuron.energy)
            reward = calculate_reward(neuron.states["novelty"], energy_loss)
            neuron.apply_reward(reward)
            neuron.finalize_tick(had_input=bool(events))

        self.queue.push_many(emitted)
        actual_state = self._aggregate_state()
        prediction_error = self._update_prediction_error(actual_state)
        snapshot = {
            "tick": tick,
            "ready_events": len(ready),
            "emitted_events": len(emitted),
            "alive_neurons": sum(1 for neuron in self.neurons.values() if neuron.alive),
            "prediction_error": prediction_error,
        }
        self.replay.append(snapshot)
        self._update_corpses()
        self.check_reproduction()
        self.check_death()
        return snapshot

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
            sum((actual_state[key] - self.expected_state.get(key, 0.0)) ** 2 for key in actual_state)
        )
        alpha = 0.3
        self.expected_state = {
            key: self.expected_state.get(key, 0.0) + alpha * (actual_state[key] - self.expected_state.get(key, 0.0))
            for key in actual_state
        }
        return prediction_error

    def check_reproduction(self) -> None:
        if len(self.neurons) >= self.max_neurons:
            return
        candidates = [neuron for neuron in self.neurons.values() if neuron.alive]
        random.shuffle(candidates)
        for neuron in candidates:
            if len(self.neurons) >= self.max_neurons:
                break
            if neuron.energy < self.reproduction_threshold:
                continue
            child = neuron.spawn(self.mutation_scale, self.reproduction_share)
            neuron.energy = max(0.0, neuron.energy - child.energy)
            self.add_neuron(child)
            self.connect(neuron.id, child.id, weight=1.0, delay=1)
            self.connect(child.id, neuron.id, weight=1.0, delay=1)
        self._prune_links()

    def check_death(self) -> None:
        if len(self.neurons) > self.max_neurons:
            ordered = sorted(self.neurons.values(), key=lambda item: item.energy)
            overflow = len(self.neurons) - self.max_neurons
            for neuron in ordered[:overflow]:
                neuron.alive = False

        dead_ids = [neuron.id for neuron in self.neurons.values() if not neuron.alive or neuron.energy <= 0.0]
        for neuron_id in dead_ids:
            neuron = self.neurons.get(neuron_id)
            if neuron is None:
                continue
            params = {
                "tau_m": neuron.tau_m,
                "r_m": neuron.r_m,
                "v_th": neuron.v_th,
                "v_rest": neuron.v_rest,
                "v_reset": neuron.v_reset,
                "spike_cost": neuron.spike_cost,
                "input_gain": neuron.input_gain,
            }
            for attr in ["a", "b", "c", "d", "v_peak", "delta_t", "tau_w"]:
                if hasattr(neuron, attr):
                    value = getattr(neuron, attr)
                    if isinstance(value, (int, float)):
                        params[attr] = float(value)
            self.corpses.append(Corpse(pos=neuron.pos, energy=neuron.energy, params=params, ttl=self.corpse_ttl))
            self.neurons.pop(neuron_id, None)

    def _update_corpses(self) -> None:
        if not self.corpses:
            return
        alive = [neuron for neuron in self.neurons.values() if neuron.alive]
        updated: list[Corpse] = []
        for corpse in self.corpses:
            corpse.ttl -= 1
            if corpse.energy > 0 and alive:
                nearest = None
                nearest_dist = None
                for neuron in alive:
                    dx = neuron.pos[0] - corpse.pos[0]
                    dy = neuron.pos[1] - corpse.pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= self.scavenger_radius and (nearest_dist is None or dist < nearest_dist):
                        nearest = neuron
                        nearest_dist = dist
                if nearest is not None:
                    nearest.energy += corpse.energy
                    corpse.energy = 0.0
            if corpse.ttl > 0 and corpse.energy > 0.0:
                updated.append(corpse)
        self.corpses = updated

    def _prune_links(self) -> None:
        total_links = sum(len(neuron.links) for neuron in self.neurons.values())
        if total_links <= self.max_links:
            return
        all_links: list[tuple[str, str]] = []
        for neuron_id, neuron in self.neurons.items():
            for target_id in neuron.links.keys():
                all_links.append((neuron_id, target_id))
        random.shuffle(all_links)
        to_remove = total_links - self.max_links
        for source_id, target_id in all_links[:to_remove]:
            if source_id in self.neurons:
                self.neurons[source_id].links.pop(target_id, None)
