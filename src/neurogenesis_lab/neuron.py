from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable
from uuid import uuid4

from neurogenesis_lab.encoding import Event


@dataclass
class Neuron:
    id: str = field(default_factory=lambda: uuid4().hex)
    pos: tuple[float, float] = (0.0, 0.0)
    energy: float = 1.0
    alive: bool = True
    links: dict[str, dict[str, float]] = field(default_factory=dict)
    mem_size: int = 64
    mem: deque[Event] = field(default_factory=lambda: deque(maxlen=64))
    transitions: Counter[tuple[int, int]] = field(default_factory=Counter)

    def receive(self, event: Event) -> list[Event]:
        if not self.alive:
            return []
        self.mem.append(event)
        self.energy = max(0.0, self.energy + 0.1)
        if len(self.mem) >= 2:
            prev = self.mem[-2]
            self.transitions[(prev.token_id, event.token_id)] += 1
        return self.emit(event)

    def emit(self, event: Event) -> list[Event]:
        emitted: list[Event] = []
        if not self.links:
            return emitted
        for neighbor_id, link in self.links.items():
            weight = float(link.get("weight", 1.0))
            delay = int(link.get("delay", 0))
            if weight <= 0:
                continue
            emitted.append(
                Event(
                    token_id=event.token_id,
                    t=event.t + max(0, delay),
                    src_id=neighbor_id,
                    meta={"weight": weight, "from": self.id},
                )
            )
        return emitted

    def move(self) -> None:
        if not self.alive:
            return
        dx = random.uniform(-1.0, 1.0)
        dy = random.uniform(-1.0, 1.0)
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        self.energy = max(0.0, self.energy - 0.01)
        if self.energy == 0.0:
            self.alive = False

    def observe(self) -> dict[str, Iterable[dict[str, int]]]:
        top = self.transitions.most_common()
        return {
            "transition_counts": [
                {"from": pair[0], "to": pair[1], "count": count} for pair, count in top
            ]
        }
