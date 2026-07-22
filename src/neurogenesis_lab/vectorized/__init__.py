"""vectorized 子包 — NumPy 加速的向量化仿真引擎。

本包由原单文件 ``vectorized.py`` 按功能拆分而来，公共 API 保持不变：

- ``engine``：``VectorizedNetwork`` 主类（组合 dynamics / propagation /
  lifecycle / persistence Mixin）。
- ``spatial``：``SpatialGrid`` 网格空间索引。
- ``state``：``VectorizedState`` 全字段容器 + 基础状态工具。
"""

from __future__ import annotations

from neurogenesis_lab.vectorized.engine import VectorizedNetwork
from neurogenesis_lab.vectorized.spatial import SpatialGrid

__all__ = [
    "SpatialGrid",
    "VectorizedNetwork",
]
