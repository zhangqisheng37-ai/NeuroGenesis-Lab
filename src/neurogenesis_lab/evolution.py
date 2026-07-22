"""演化策略 — 引擎无关的纯决策函数（行为保持）。

与 ``adaptive.py`` 同属策略层：把 ``Network`` 与 ``VectorizedNetwork`` 共享的演化
决策集中于此，便于统一维护与单元测试，避免两套实现产生行为漂移。

由于两引擎数据布局不同（Network 的对象字典 vs Vectorized 的 NumPy 数组），
部分策略提供“标量版 + 数组版”成对函数，二者复刻各自引擎的现有精确行为。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neurogenesis_lab.config import ENDANGERED_EXIT_MULTIPLIER


def update_endangered_mode(
    current_mode: bool, alive_count: int, threshold: int
) -> bool:
    """根据存活数更新濒危模式标志（两引擎共享）。

    复刻现有两条独立判定：存活数 <= 阈值且当前非濒危则进入；
    存活数 > 阈值 * ENDANGERED_EXIT_MULTIPLIER 则退出；否则保持不变。

    Args:
        current_mode: 当前濒危模式标志。
        alive_count: 当前存活神经元数量。
        threshold: 濒危告警阈值。

    Returns:
        更新后的濒危模式标志。
    """
    new_mode = current_mode
    if alive_count <= threshold and not current_mode:
        new_mode = True
    if alive_count > threshold * ENDANGERED_EXIT_MULTIPLIER:
        new_mode = False
    return new_mode


def select_rescued(
    candidates: Sequence[tuple[str, float]], deficit: int
) -> list[str]:
    """从死亡候选中挑选待复活者（Network 标量路径）。

    按能量降序稳定排序取前 ``deficit`` 个，复刻 Network 现有 ``list.sort`` 行为。

    Args:
        candidates: (neuron_id, energy) 序列。
        deficit: 需复活的数量（min_neurons - alive_count）。

    Returns:
        待复活的 neuron_id 列表；deficit <= 0 时返回空列表。
    """
    if deficit <= 0:
        return []
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    return [key for key, _ in ranked[:deficit]]


def select_rescued_indices(
    energies: np.ndarray, dead_indices: np.ndarray, deficit: int
) -> np.ndarray:
    """从死亡索引中挑选待复活者（Vectorized 数组路径）。

    使用 ``np.argsort(-energies[dead_indices])`` 取能量最高的前 ``deficit`` 个，
    保留 argsort 以复刻 Vectorized 现有并列次序。

    Args:
        energies: 全体能量数组。
        dead_indices: 本 tick 死亡神经元的索引数组。
        deficit: 需复活的数量（min_neurons - alive_after）。

    Returns:
        待复活的索引数组；deficit <= 0 或无死亡时返回空数组。
    """
    if deficit <= 0 or len(dead_indices) == 0:
        return np.empty(0, dtype=dead_indices.dtype)
    ordered = dead_indices[np.argsort(-energies[dead_indices])]
    return ordered[:deficit]


def can_reproduce(energy: float, threshold: float) -> bool:
    """判定单个神经元是否达到繁殖能量阈值（Network 标量路径）。

    Args:
        energy: 神经元当前能量。
        threshold: 繁殖能量阈值。

    Returns:
        能量 >= 阈值 时返回 True。
    """
    return energy >= threshold


def reproduction_mask(
    alive: np.ndarray, energy: np.ndarray, threshold: float
) -> np.ndarray:
    """计算可繁殖掩码（Vectorized 数组路径）：存活且能量达阈值。

    Args:
        alive: 存活掩码数组。
        energy: 能量数组。
        threshold: 繁殖能量阈值。

    Returns:
        布尔掩码数组，标记可繁殖的神经元。
    """
    return alive & (energy >= threshold)


def child_energy(parent_energy: float, share: float) -> float:
    """按分享比例派生子代初始能量（两引擎共享的能量派生规则）。

    Args:
        parent_energy: 母体当前能量。
        share: 能量分享比例，范围 (0, 1]。

    Returns:
        子代获得的初始能量 = parent_energy * share。
    """
    return parent_energy * share


def decay_corpse_energy(energy: float, rate: float) -> float:
    """尸体能量自然衰减（下限 0）— 尸体生命周期共享规则。

    Args:
        energy: 尸体当前能量。
        rate: 每 tick 衰减量。

    Returns:
        衰减后的能量，不低于 0。
    """
    return max(0.0, energy - rate)


def is_corpse_expired(ttl: int, energy: float) -> bool:
    """判定尸体是否应移除：TTL 归零或能量耗尽。

    Args:
        ttl: 尸体剩余存活 tick 数。
        energy: 尸体当前能量。

    Returns:
        ttl <= 0 或 energy <= 0 时返回 True。
    """
    return ttl <= 0 or energy <= 0.0
