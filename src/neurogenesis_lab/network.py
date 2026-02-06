from __future__ import annotations

import heapq
import math
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
class Network:
    neurons: dict[str, Neuron]
    queue: EventQueue = field(default_factory=EventQueue)
    replay: list[dict[str, float | int]] = field(default_factory=list)
    expected_state: dict[str, float] | None = None

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
                input_current += float(event.meta.get("weight", 1.0))
            spiked = neuron.update(dt, input_current)
            if spiked and neuron.last_event is not None:
                emitted.extend(neuron.emit(neuron.last_event))
            neuron.move()
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
        return None

    def check_death(self) -> None:
        return None
