"""自适应机制 — 网络/向量化引擎共享的自适应衰减和繁殖逻辑。"""

from __future__ import annotations

from neurogenesis_lab.config import (
    DENSITY_HIGH_THRESHOLD,
    DENSITY_LOW_THRESHOLD,
    DECAY_HIGH_MULTIPLIER,
    DECAY_LOW_MULTIPLIER,
    POPULATION_HIGH_THRESHOLD,
    POPULATION_LOW_THRESHOLD,
    REPRO_SHARE_HIGH_MULTIPLIER,
    REPRO_SHARE_LOW_MULTIPLIER,
    REPRO_SHARE_MAX,
    REPRO_SHARE_MIN,
)


def calculate_adaptive_decay(
    enabled: bool,
    base_decay: float,
    alive_count: int,
    max_neurons: int,
) -> float:
    """根据种群密度计算自适应能量衰减系数。

    Args:
        enabled: 是否启用自适应衰减。
        base_decay: 基础能量衰减量。
        alive_count: 当前存活神经元数量。
        max_neurons: 最大神经元容量。

    Returns:
        调整后的能量衰减值。
    """
    if not enabled:
        return base_decay

    # 种群密度比例
    density_ratio = alive_count / max_neurons if max_neurons > 0 else 0.5

    if density_ratio > DENSITY_HIGH_THRESHOLD:
        # 过度拥挤，增加衰减
        return base_decay * DECAY_HIGH_MULTIPLIER
    elif density_ratio < DENSITY_LOW_THRESHOLD:
        # 种群稀少，减少衰减
        return base_decay * DECAY_LOW_MULTIPLIER
    else:
        return base_decay


def calculate_adaptive_reproduction_share(
    enabled: bool,
    base_share: float,
    alive_count: int,
    max_neurons: int,
) -> float:
    """根据种群规模计算自适应繁殖能量分享比例。

    Args:
        enabled: 是否启用自适应繁殖。
        base_share: 基础繁殖能量分享比例。
        alive_count: 当前存活神经元数量。
        max_neurons: 最大神经元容量。

    Returns:
        调整后的繁殖能量分享比例。
    """
    if not enabled:
        return base_share

    # 种群规模比例
    population_ratio = alive_count / max_neurons if max_neurons > 0 else 0.5

    if population_ratio > POPULATION_HIGH_THRESHOLD:
        # 接近容量上限，增加成本（母体分享更多能量）
        return min(REPRO_SHARE_MAX, base_share * REPRO_SHARE_HIGH_MULTIPLIER)
    elif population_ratio < POPULATION_LOW_THRESHOLD:
        # 种群稀少，降低成本（鼓励繁殖）
        return max(REPRO_SHARE_MIN, base_share * REPRO_SHARE_LOW_MULTIPLIER)
    else:
        return base_share
