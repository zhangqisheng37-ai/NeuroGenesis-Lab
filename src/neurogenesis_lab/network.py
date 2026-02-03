from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable

from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import Neuron


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
    replay: list[dict[str, int]] = field(default_factory=list)

    def connect(self, source_id: str, target_id: str, weight: float = 1.0, delay: int = 0) -> None:
        neuron = self.neurons[source_id]
        neuron.links[target_id] = {"weight": weight, "delay": delay}

    def add_neuron(self, neuron: Neuron) -> None:
        self.neurons[neuron.id] = neuron

    def enqueue(self, events: Iterable[Event]) -> None:
        self.queue.push_many(events)

    def tick(self, tick: int) -> dict[str, int]:
        ready = self.queue.pop_ready(tick)
        emitted: list[Event] = []
        for event in ready:
            if event.src_id not in self.neurons:
                continue
            emitted.extend(self.neurons[event.src_id].receive(event))
        self.queue.push_many(emitted)
        for neuron in self.neurons.values():
            neuron.move()
        snapshot = {
            "tick": tick,
            "ready_events": len(ready),
            "emitted_events": len(emitted),
            "alive_neurons": sum(1 for neuron in self.neurons.values() if neuron.alive),
        }
        self.replay.append(snapshot)
        return snapshot
