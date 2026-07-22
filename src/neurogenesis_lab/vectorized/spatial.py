"""网格空间索引 — vectorized 子包的近邻查找部分。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpatialGrid:
    """简单网格空间索引 — 加速近邻查找。

    将二维空间划分为固定大小的网格单元，支持快速半径查询。

    Attributes:
        cell_size: 网格单元边长，默认 5.0。
        _grid: 内部网格字典，键为 (cx, cy) 网格坐标，值为神经元索引列表。
    """

    cell_size: float = 5.0
    _grid: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def _cell_key(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(math.floor(x / self.cell_size)),
            int(math.floor(y / self.cell_size)),
        )

    def build(self, positions: np.ndarray) -> None:
        """从位置数组重建空间索引。

        Args:
            positions: 形状为 (N, 2) 的位置数组。
        """
        self._grid.clear()
        for i in range(len(positions)):
            key = self._cell_key(float(positions[i, 0]), float(positions[i, 1]))
            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(i)

    def query_radius(self, x: float, y: float, radius: float, positions: np.ndarray) -> list[int]:
        """查找指定坐标半径内的所有神经元索引。

        Args:
            x: 查询中心 x 坐标。
            y: 查询中心 y 坐标。
            radius: 搜索半径。
            positions: 形状为 (N, 2) 的位置数组，用于精确距离过滤。

        Returns:
            半径内的神经元索引列表。
        """
        cx, cy = self._cell_key(x, y)
        r_cells = int(math.ceil(radius / self.cell_size))
        candidates: list[int] = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                key = (cx + dx, cy + dy)
                if key in self._grid:
                    candidates.extend(self._grid[key])
        # 精确距离过滤
        result: list[int] = []
        r2 = radius * radius
        for idx in candidates:
            ddx = float(positions[idx, 0]) - x
            ddy = float(positions[idx, 1]) - y
            if ddx * ddx + ddy * ddy <= r2:
                result.append(idx)
        return result
