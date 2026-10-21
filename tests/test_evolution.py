"""evolution 策略与回归测试 — 纯策略函数边界 + 标量与向量化一致性（原两个文件合并）"""

from __future__ import annotations

import numpy as np
import pytest

from neurogenesis_lab.config import ENDANGERED_EXIT_MULTIPLIER, seed_all
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.evolution import (
    can_reproduce,
    child_energy,
    decay_corpse_energy,
    is_corpse_expired,
    reproduction_mask,
    select_rescued,
    select_rescued_indices,
    update_endangered_mode,
)
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.vectorized import VectorizedNetwork


class TestUpdateEndangeredMode:
    """濒危模式转移逻辑。"""

    def test_enter_when_at_or_below_threshold(self) -> None:
        assert update_endangered_mode(False, 5, 10) is True
        assert update_endangered_mode(False, 10, 10) is True  # 边界：等于阈值也进入

    def test_hold_in_hysteresis_band(self) -> None:
        # 阈值 10，退出阈值 = 10 * ENDANGERED_EXIT_MULTIPLIER = 15；alive=12 两条 if 均不触发
        assert ENDANGERED_EXIT_MULTIPLIER == 1.5
        assert update_endangered_mode(False, 12, 10) is False
        assert update_endangered_mode(True, 12, 10) is True

    def test_exit_above_exit_threshold(self) -> None:
        assert update_endangered_mode(True, 16, 10) is False

    def test_already_endangered_stays_when_below(self) -> None:
        assert update_endangered_mode(True, 3, 10) is True


class TestSelectRescued:
    """Network 标量救援选择。"""

    def test_deficit_non_positive_returns_empty(self) -> None:
        assert select_rescued([("a", 1.0)], 0) == []
        assert select_rescued([("a", 1.0)], -1) == []

    def test_picks_highest_energy(self) -> None:
        candidates = [("a", 0.1), ("b", 0.9), ("c", 0.5)]
        assert select_rescued(candidates, 2) == ["b", "c"]

    def test_stable_order_on_ties(self) -> None:
        candidates = [("a", 0.5), ("b", 0.5), ("c", 0.5)]
        assert select_rescued(candidates, 2) == ["a", "b"]

    def test_deficit_exceeds_candidates(self) -> None:
        assert select_rescued([("a", 0.5), ("b", 0.9)], 5) == ["b", "a"]

    def test_empty_candidates(self) -> None:
        assert select_rescued([], 3) == []


class TestSelectRescuedIndices:
    """Vectorized 数组救援选择。"""

    def test_deficit_non_positive_returns_empty(self) -> None:
        result = select_rescued_indices(np.array([1.0, 2.0]), np.array([0, 1]), 0)
        assert list(result) == []

    def test_empty_dead_returns_empty(self) -> None:
        result = select_rescued_indices(
            np.array([1.0]), np.array([], dtype=np.int64), 2
        )
        assert list(result) == []

    def test_picks_highest_energy(self) -> None:
        energies = np.array([0.1, 0.9, 0.5, 0.2])
        dead = np.array([0, 2, 3])  # 对应能量 0.1 / 0.5 / 0.2
        result = select_rescued_indices(energies, dead, 2)
        assert list(result) == [2, 3]

    def test_deficit_exceeds_dead(self) -> None:
        energies = np.array([0.3, 0.7])
        result = select_rescued_indices(energies, np.array([0, 1]), 5)
        assert list(result) == [1, 0]


class TestReproductionPolicy:
    """繁殖资格与子代能量派生。"""

    def test_can_reproduce_boundary(self) -> None:
        assert can_reproduce(2.5, 2.5) is True  # 边界：等于阈值可繁殖
        assert can_reproduce(2.49, 2.5) is False

    def test_reproduction_mask(self) -> None:
        alive = np.array([True, True, False, True])
        energy = np.array([3.0, 1.0, 5.0, 2.5])
        result = reproduction_mask(alive, energy, 2.5)
        assert list(result) == [True, False, False, True]

    def test_child_energy(self) -> None:
        assert child_energy(2.0, 0.4) == 0.8


class TestCorpseLifecycle:
    """尸体能量衰减与过期判定。"""

    def test_decay_clamps_at_zero(self) -> None:
        assert decay_corpse_energy(0.05, 0.02) == pytest.approx(0.03)
        assert decay_corpse_energy(0.01, 0.02) == 0.0

    def test_is_corpse_expired(self) -> None:
        assert is_corpse_expired(0, 1.0) is True
        assert is_corpse_expired(5, 0.0) is True
        assert is_corpse_expired(5, 1.0) is False


# ======================================================================
# 合并自 test_evolution_regression.py：演化回归安全网 — 锁定两引擎在固定种子下的逐 tick 演化轨迹。
# ======================================================================


TrajRow = tuple[int, int, int, int, float]

NETWORK_SEED = 12345
VECTORIZED_SEED = 2024
TICKS = 80

_SUMMARY_INT_KEYS = (
    "total_neurons",
    "alive_neurons",
    "total_spikes",
    "total_propagations",
    "birth_count",
    "death_count",
)


def _summary_ints(engine: object) -> dict[str, int]:
    summary = engine.get_summary()  # type: ignore[attr-defined]
    return {key: int(summary[key]) for key in _SUMMARY_INT_KEYS}


def _row(snapshot: dict) -> TrajRow:
    return (
        int(snapshot["alive_neurons"]),
        int(snapshot["births"]),
        int(snapshot["deaths"]),
        int(snapshot["tick_spikes"]),
        round(float(snapshot["mean_energy"]), 6),
    )


def _run_network() -> tuple[list[TrajRow], dict[str, int]]:
    """固定 Network 场景：环形 + 近邻连接 + 周期强刺激，覆盖繁殖/死亡/尸体/濒危。"""
    seed_all(NETWORK_SEED)
    neurons: dict[str, LIFNeuron] = {}
    for i in range(10):
        neuron = LIFNeuron(id=f"n{i}", energy=3.0, pos=(float(i), 0.0))
        neurons[neuron.id] = neuron
    net = Network(
        neurons=neurons,
        reproduction_threshold=2.5,
        reproduction_share=0.4,
        mutation_scale=0.05,
        energy_decay=0.03,
        max_neurons=25,
        max_links=200,
        corpse_ttl=15,
        scavenger_radius=2.5,
        input_current_scale=1.5,
        move_cost=0.01,
        energy_cap=5.0,
        endangered_threshold=6,
        min_neurons=2,
    )
    ids = list(neurons)
    for i, source in enumerate(ids):
        net.connect(source, ids[(i + 1) % len(ids)], weight=1.0, delay=1)
    net.connect_by_proximity(radius=5.0, max_links_per_neuron=3)
    for t0 in range(1, TICKS + 1, 5):
        net.enqueue([Event(token_id=49, t=t0, src_id="n0", meta={"weight": 40.0})])
    traj: list[TrajRow] = []
    for t in range(1, TICKS + 1):
        traj.append(_row(net.tick(t, dt=1.0)))
    return traj, _summary_ints(net)


def _run_vectorized() -> tuple[list[TrajRow], dict[str, int]]:
    """固定 VectorizedNetwork 场景：初始高能量触发繁殖，衰减触发死亡/救援/尸体。"""
    seed_all(VECTORIZED_SEED)
    vnet = VectorizedNetwork(
        reproduction_threshold=2.5,
        reproduction_share=0.4,
        mutation_scale=0.05,
        energy_decay=0.05,
        max_neurons=40,
        move_cost=0.01,
        energy_cap=5.0,
        endangered_threshold=8,
        min_neurons=2,
        corpse_ttl=15,
        scavenger_radius=2.5,
        input_current_scale=1.0,
    )
    vnet.init_neurons(20, energy_init=3.0, spread=8.0)
    vnet.connect_knn(k=4)
    # 预留空槽（非尸体）以触发繁殖：后 8 个槽位标记为非存活，供高能量存活者复用。
    vnet.alive[12:] = False
    traj: list[TrajRow] = []
    for t in range(1, TICKS + 1):
        traj.append(_row(vnet.tick(t, dt=1.0)))
    return traj, _summary_ints(vnet)


# ── 黄金值（在未改动代码上采集后内嵌） ─────────────────────────────
EXPECTED_NETWORK: list[TrajRow] = [
    (10, 10, 0, 0, 2.965),
    (20, 0, 0, 0, 1.442469),
    (20, 0, 0, 0, 1.402438),
    (20, 0, 0, 0, 1.362407),
    (20, 0, 0, 0, 1.322376),
    (20, 0, 0, 0, 1.284845),
    (20, 0, 0, 0, 1.244814),
    (20, 0, 0, 0, 1.204783),
    (20, 0, 0, 0, 1.164752),
    (20, 0, 0, 0, 1.124721),
    (20, 0, 0, 0, 1.08719),
    (20, 0, 0, 0, 1.047159),
    (20, 0, 0, 0, 1.007128),
    (20, 0, 0, 0, 0.967097),
    (20, 0, 0, 0, 0.927066),
    (20, 0, 0, 0, 0.889535),
    (20, 0, 0, 0, 0.849504),
    (20, 0, 0, 0, 0.809473),
    (20, 0, 0, 0, 0.769442),
    (20, 0, 0, 0, 0.729411),
    (20, 0, 0, 0, 0.69188),
    (20, 0, 0, 0, 0.651849),
    (20, 0, 0, 0, 0.611818),
    (20, 0, 0, 0, 0.571787),
    (20, 0, 0, 0, 0.531756),
    (20, 0, 0, 1, 0.491725),
    (20, 0, 0, 0, 0.456569),
    (20, 0, 0, 0, 0.419038),
    (20, 0, 0, 0, 0.381507),
    (20, 0, 0, 0, 0.343976),
    (20, 0, 9, 0, 0.313916),
    (11, 0, 1, 0, 0.530909),
    (10, 0, 0, 0, 0.544),
    (10, 0, 0, 0, 0.504),
    (10, 0, 0, 0, 0.464),
    (10, 0, 0, 0, 0.429),
    (10, 0, 0, 0, 0.389),
    (10, 0, 0, 0, 0.349),
    (10, 0, 0, 0, 0.309),
    (10, 0, 0, 0, 0.269),
    (10, 0, 0, 0, 0.234),
    (10, 0, 0, 0, 0.194),
    (10, 0, 0, 0, 0.154),
    (10, 0, 0, 0, 0.114),
    (10, 0, 0, 0, 0.074),
    (10, 0, 5, 0, 0.051),
    (5, 0, 0, 0, 0.0815),
    (1, 0, 3, 0, 0.365),
    (2, 0, 0, 0, 0.212),
    (2, 0, 0, 0, 0.1915),
    (2, 0, 0, 0, 0.196),
    (2, 0, 0, 0, 0.1755),
    (2, 0, 0, 0, 0.15625),
    (2, 0, 0, 0, 0.18575),
    (2, 0, 0, 0, 0.16525),
    (2, 0, 0, 1, 0.14475),
    (2, 0, 0, 0, 0.14925),
    (2, 0, 0, 0, 0.12875),
    (2, 0, 0, 0, 0.10825),
    (2, 0, 0, 0, 0.08775),
    (1, 0, 0, 0, 0.1985),
    (2, 0, 0, 0, 0.12875),
    (2, 0, 0, 0, 0.10825),
    (2, 0, 0, 0, 0.08775),
    (2, 0, 0, 0, 0.06725),
    (2, 0, 0, 0, 0.073),
    (2, 0, 0, 0, 0.1025),
    (2, 0, 0, 0, 0.082),
    (2, 0, 0, 0, 0.0615),
    (2, 0, 0, 0, 0.041),
    (2, 0, 0, 0, 0.04675),
    (2, 0, 0, 0, 0.07625),
    (2, 0, 0, 0, 0.05575),
    (2, 0, 0, 0, 0.03525),
    (2, 0, 0, 0, 0.01475),
    (2, 0, 0, 0, 0.0205),
    (2, 0, 0, 0, 0.05),
    (2, 0, 0, 0, 0.0295),
    (2, 0, 0, 0, 0.059),
    (2, 0, 0, 0, 0.0385),
]
EXPECTED_NETWORK_SUMMARY: dict[str, int] = {
    "total_neurons": 2,
    "alive_neurons": 2,
    "total_spikes": 2,
    "total_propagations": 10,
    "birth_count": 10,
    "death_count": 18,
}
EXPECTED_VECTORIZED: list[TrajRow] = [
    (20, 8, 0, 0, 1.764),
    (20, 0, 0, 0, 1.704),
    (20, 0, 0, 0, 1.644),
    (20, 0, 0, 0, 1.584),
    (20, 0, 0, 0, 1.524),
    (20, 0, 0, 0, 1.464),
    (20, 0, 0, 0, 1.404),
    (20, 0, 0, 0, 1.344),
    (20, 0, 0, 0, 1.284),
    (20, 0, 0, 0, 1.224),
    (20, 0, 0, 0, 1.164),
    (20, 0, 0, 0, 1.104),
    (20, 0, 0, 0, 1.044),
    (20, 0, 0, 0, 0.984),
    (20, 0, 0, 0, 0.924),
    (20, 0, 0, 0, 0.864),
    (20, 0, 0, 0, 0.804),
    (20, 0, 0, 0, 0.744),
    (20, 0, 0, 0, 0.684),
    (20, 0, 0, 0, 0.624),
    (12, 0, 8, 0, 0.956),
    (12, 0, 0, 0, 0.896),
    (12, 0, 0, 0, 0.836),
    (12, 0, 0, 0, 0.776),
    (12, 0, 0, 0, 0.716),
    (12, 0, 0, 0, 0.656),
    (12, 0, 0, 0, 0.596),
    (12, 0, 0, 0, 0.536),
    (12, 0, 0, 0, 0.476),
    (12, 0, 0, 0, 0.416),
    (4, 0, 8, 0, 1.14),
    (4, 0, 0, 0, 1.1125),
    (4, 0, 0, 0, 1.085),
    (4, 0, 0, 0, 1.0575),
    (4, 0, 0, 0, 1.03),
    (4, 0, 0, 0, 1.0025),
    (4, 0, 0, 0, 0.975),
    (4, 0, 0, 0, 0.9475),
    (4, 0, 0, 0, 0.92),
    (4, 0, 0, 0, 0.8925),
    (4, 0, 0, 0, 0.865),
    (4, 0, 0, 0, 0.8375),
    (4, 0, 0, 0, 0.81),
    (4, 0, 0, 0, 0.7825),
    (4, 0, 0, 0, 0.755),
    (4, 0, 0, 0, 0.7275),
    (4, 0, 0, 0, 0.7),
    (4, 0, 0, 0, 0.6725),
    (4, 0, 0, 0, 0.645),
    (4, 0, 0, 0, 0.6175),
    (4, 0, 0, 0, 0.59),
    (4, 0, 0, 0, 0.5625),
    (4, 0, 0, 0, 0.535),
    (4, 0, 0, 0, 0.5075),
    (4, 0, 0, 0, 0.48),
    (4, 0, 0, 0, 0.4525),
    (4, 0, 0, 0, 0.425),
    (4, 0, 0, 0, 0.3975),
    (4, 0, 0, 0, 0.37),
    (4, 0, 0, 0, 0.3425),
    (4, 0, 0, 0, 0.315),
    (4, 0, 0, 0, 0.2875),
    (4, 0, 0, 0, 0.26),
    (4, 0, 0, 0, 0.2325),
    (4, 0, 0, 0, 0.205),
    (4, 0, 0, 0, 0.1775),
    (4, 0, 0, 0, 0.15),
    (4, 0, 0, 0, 0.1225),
    (4, 0, 0, 0, 0.095),
    (4, 0, 0, 0, 0.0675),
    (4, 0, 0, 0, 0.04),
    (4, 0, 0, 0, 0.0125),
    (2, 0, 2, 0, 0.1),
    (2, 0, 0, 0, 0.0725),
    (2, 0, 0, 0, 0.045),
    (2, 0, 0, 0, 0.0175),
    (2, 0, 0, 0, 0.1),
    (2, 0, 0, 0, 0.0725),
    (2, 0, 0, 0, 0.045),
    (2, 0, 0, 0, 0.0175),
]
EXPECTED_VECTORIZED_SUMMARY: dict[str, int] = {
    "total_neurons": 20,
    "alive_neurons": 2,
    "total_spikes": 0,
    "total_propagations": 0,
    "birth_count": 8,
    "death_count": 18,
}


def test_network_evolution_golden_trajectory() -> None:
    traj, summary = _run_network()
    assert traj == EXPECTED_NETWORK
    assert summary == EXPECTED_NETWORK_SUMMARY


def test_vectorized_evolution_golden_trajectory() -> None:
    traj, summary = _run_vectorized()
    assert traj == EXPECTED_VECTORIZED
    assert summary == EXPECTED_VECTORIZED_SUMMARY


def _emit(name: str, traj: list[TrajRow]) -> str:
    lines = [f"{name} = ["]
    lines.extend(f"    {row!r}," for row in traj)
    lines.append("]")
    return "\n".join(lines)


if __name__ == "__main__":
    ntraj, nsum = _run_network()
    vtraj, vsum = _run_vectorized()
    print(_emit("EXPECTED_NETWORK", ntraj))
    print(f"EXPECTED_NETWORK_SUMMARY = {nsum!r}")
    print(_emit("EXPECTED_VECTORIZED", vtraj))
    print(f"EXPECTED_VECTORIZED_SUMMARY = {vsum!r}")
