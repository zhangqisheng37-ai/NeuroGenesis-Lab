"""evolution.py 纯策略函数单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from neurogenesis_lab.config import ENDANGERED_EXIT_MULTIPLIER
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
