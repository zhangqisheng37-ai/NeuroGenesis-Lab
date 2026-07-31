"""单元测试 — 实时可视化 WebSocket 协议（消息构建、命令解析、放电追踪）"""

from __future__ import annotations

from typing import Any

import pytest

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.live_protocol import (
    CMD_PAUSE,
    CMD_SPEED,
    MAX_TICK_MS,
    MSG_INIT,
    MSG_STATUS,
    MSG_TICK,
    STATE_RUNNING,
    TickMessageBuilder,
    build_init_message,
    build_status_message,
    clamp_tick_ms,
    parse_command,
)


class FakeNeuron:
    """轻量神经元替身，仅提供协议构建所需属性。"""

    def __init__(
        self,
        nid: str,
        pos: tuple[float, float],
        energy: float,
        alive: bool,
        spike_count: int,
        links: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.id = nid
        self.pos = pos
        self.energy = energy
        self.alive = alive
        self.spike_count = spike_count
        self.links = links or {}


class FakeNetwork:
    """轻量网络替身，提供 neurons 映射。"""

    def __init__(self, neurons: list[FakeNeuron]) -> None:
        self.neurons = {n.id: n for n in neurons}


# ── 消息构建 ────────────────────────────────────────────────


def test_build_init_message() -> None:
    msg = build_init_message({"ticks": 100}, total_neurons=8)
    assert msg["type"] == MSG_INIT
    assert msg["schema_version"] == SCHEMA_VERSION
    assert msg["config"] == {"ticks": 100}
    assert msg["total_neurons"] == 8


def test_build_status_message() -> None:
    msg = build_status_message(STATE_RUNNING, tick=5, tick_ms=50)
    assert msg["type"] == MSG_STATUS
    assert msg["state"] == STATE_RUNNING
    assert msg["tick"] == 5
    assert msg["tick_ms"] == 50


# ── 调速裁剪 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"),
    [(50, 50), (-10, 0), (99999, MAX_TICK_MS), ("30", 30), ("bad", 0), (None, 0)],
)
def test_clamp_tick_ms(value: Any, expected: int) -> None:
    assert clamp_tick_ms(value) == expected


# ── 命令解析 ────────────────────────────────────────────────


def test_parse_command_valid() -> None:
    assert parse_command({"type": CMD_PAUSE}) == (CMD_PAUSE, {})
    assert parse_command({"type": CMD_SPEED, "tick_ms": 100}) == (CMD_SPEED, {"tick_ms": 100})


@pytest.mark.parametrize(
    "raw",
    [{"type": "unknown"}, {"no_type": 1}, "not-a-dict", 42, {"type": 123}],
)
def test_parse_command_invalid(raw: Any) -> None:
    with pytest.raises(ValueError):
        parse_command(raw)


# ── tick 消息与放电追踪 ─────────────────────────────────────


def test_tick_message_builder_basic() -> None:
    net = FakeNetwork(
        [
            FakeNeuron("a", (0.0, 0.0), 1.5, True, 1, links={"b": {"weight": 2.0, "delay": 1}}),
            FakeNeuron("b", (1.0, 2.0), 0.5, True, 0),
        ]
    )
    builder = TickMessageBuilder()
    snapshot = {"tick": 1, "alive_neurons": 2, "total_neurons": 2, "tick_spikes": 1}
    msg = builder.build(1, snapshot, net)

    assert msg["type"] == MSG_TICK
    assert msg["tick"] == 1
    assert len(msg["neurons"]) == 2
    node_a = next(n for n in msg["neurons"] if n["id"] == "a")
    assert node_a["x"] == 0.0 and node_a["y"] == 0.0
    assert node_a["spiked"] is True  # spike_count 1 > prev 0
    # 连接仅来自存活神经元 a
    assert msg["connections"] == [{"source": "a", "target": "b", "weight": 2.0}]
    assert msg["stats"]["alive_neurons"] == 2


def test_tick_message_builder_spike_delta() -> None:
    neuron = FakeNeuron("a", (0.0, 0.0), 1.0, True, 0)
    net = FakeNetwork([neuron])
    builder = TickMessageBuilder()

    assert builder.build(1, {}, net)["neurons"][0]["spiked"] is False  # 0 == prev 0
    neuron.spike_count = 3
    assert builder.build(2, {}, net)["neurons"][0]["spiked"] is True  # 3 > 0
    assert builder.build(3, {}, net)["neurons"][0]["spiked"] is False  # 3 == 3


def test_tick_message_builder_dead_neuron_no_connections() -> None:
    net = FakeNetwork([FakeNeuron("a", (0.0, 0.0), 0.0, False, 0, links={"b": {"weight": 1.0}})])
    msg = TickMessageBuilder().build(1, {}, net)
    assert msg["neurons"][0]["alive"] is False
    assert msg["connections"] == []  # 死亡神经元不产出连接


def test_tick_message_builder_max_connections_cap() -> None:
    links = {f"t{i}": {"weight": 1.0} for i in range(10)}
    net = FakeNetwork([FakeNeuron("a", (0.0, 0.0), 1.0, True, 0, links=links)])
    msg = TickMessageBuilder(max_connections=3).build(1, {}, net)
    assert len(msg["connections"]) == 3


def test_tick_message_stats_defaults_for_missing_fields() -> None:
    net = FakeNetwork([FakeNeuron("a", (0.0, 0.0), 1.0, True, 0)])
    stats = TickMessageBuilder().build(1, {}, net)["stats"]
    assert stats["rule_count"] == 0
    assert stats["prediction_accuracy"] == 0.0
    assert stats["mean_energy"] == 0.0
