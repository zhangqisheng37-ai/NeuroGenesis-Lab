from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable
from uuid import uuid4

from neurogenesis_lab.encoding import Event


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def calculate_reward(novelty: float, energy_loss: float) -> float:
    return novelty - energy_loss


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
    states: dict[str, float] = field(
        default_factory=lambda: {"energy": 1.0, "entropy": 0.0, "emotion": 0.0, "novelty": 0.0}
    )
    last_energy: float = 1.0
    last_event: Event | None = None
    last_reward: float = 0.0
    v: float = -65.0
    u: float = 0.0
    w: float = 0.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_th: float = -50.0
    tau_m: float = 20.0
    r_m: float = 1.0
    spike_cost: float = 0.05
    input_gain: float = 0.05

    def __post_init__(self) -> None:
        self.mem = deque(self.mem, maxlen=self.mem_size)
        self.states["energy"] = self.energy
        self.last_energy = self.energy

    def start_tick(self) -> None:
        self.last_energy = self.energy

    def receive(self, event: Event) -> list[Event]:
        if not self.alive:
            return []
        recent_tokens = [item.token_id for item in list(self.mem)[-8:]]
        self.mem.append(event)
        self.last_event = event
        self.energy = max(0.0, self.energy + self.input_gain)
        if len(self.mem) >= 2:
            prev = self.mem[-2]
            self.transitions[(prev.token_id, event.token_id)] += 1
        if event.token_id in recent_tokens:
            self.states["novelty"] = max(0.0, self.states["novelty"] - 0.02)
        else:
            self.states["novelty"] = min(1.0, self.states["novelty"] + 0.1)
        return []

    def update(self, dt: float, input_current: float) -> bool:
        dv = (-(self.v - self.v_rest) + self.r_m * input_current) / self.tau_m
        self.v += dv * dt
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.on_spike()
            return True
        return False

    def on_spike(self) -> None:
        self.energy = max(0.0, self.energy - self.spike_cost)

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

    def apply_reward(self, reward: float) -> None:
        self.last_reward = reward
        self.states["emotion"] = clamp(self.states["emotion"] + reward * 0.5, 0.0, 1.0)

    def finalize_tick(self, had_input: bool) -> None:
        self.states["energy"] = self.energy
        if not had_input:
            self.states["novelty"] *= 0.98
            self.states["emotion"] *= 0.98
        tokens = [item.token_id for item in self.mem]
        if not tokens:
            self.states["entropy"] = 0.0
            return
        counts = Counter(tokens)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            ratio = count / total
            entropy -= ratio * math.log(ratio)
        max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
        self.states["entropy"] = entropy / max_entropy if max_entropy > 0 else 0.0

    def observe(self) -> dict[str, Iterable[dict[str, int]]]:
        top = self.transitions.most_common()
        return {
            "transition_counts": [
                {"from": pair[0], "to": pair[1], "count": count} for pair, count in top
            ],
            "states": dict(self.states),
        }


@dataclass
class LIFNeuron(Neuron):
    def update(self, dt: float, input_current: float) -> bool:
        dv = (-(self.v - self.v_rest) + self.r_m * input_current) / self.tau_m
        self.v += dv * dt
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.on_spike()
            return True
        return False


@dataclass
class IzhikevichNeuron(Neuron):
    a: float = 0.02
    b: float = 0.2
    c: float = -65.0
    d: float = 8.0
    v_peak: float = 30.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.u == 0.0:
            self.u = self.b * self.v

    def update(self, dt: float, input_current: float) -> bool:
        dv = 0.04 * self.v * self.v + 5 * self.v + 140 - self.u + input_current
        du = self.a * (self.b * self.v - self.u)
        self.v += dv * dt
        self.u += du * dt
        if self.v >= self.v_peak:
            self.v = self.c
            self.u += self.d
            self.on_spike()
            return True
        return False


@dataclass
class AdExNeuron(Neuron):
    delta_t: float = 2.0
    a: float = 1.0
    b: float = 0.5
    tau_w: float = 80.0
    v_peak: float = -40.0

    def update(self, dt: float, input_current: float) -> bool:
        exp_term = self.delta_t * math.exp((self.v - self.v_th) / self.delta_t)
        dv = (-(self.v - self.v_rest) + exp_term - self.w + input_current) / self.tau_m
        dw = (self.a * (self.v - self.v_rest) - self.w) / self.tau_w
        self.v += dv * dt
        self.w += dw * dt
        if self.v >= self.v_peak:
            self.v = self.v_reset
            self.w += self.b
            self.on_spike()
            return True
        return False
