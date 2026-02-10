"""向量化仿真引擎 — NumPy 加速，适用于万级神经元

接口兼容原有 Network，但内部状态用 NumPy 数组存储，
批量计算 LIF 动力学、空间索引、能量衰减等。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import safe_float


@dataclass
class SpatialGrid:
    """简单网格空间索引 — 加速近邻查找"""
    cell_size: float = 5.0
    _grid: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def _cell_key(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size)))

    def build(self, positions: np.ndarray) -> None:
        """从 (N, 2) 位置数组构建索引"""
        self._grid.clear()
        for i in range(len(positions)):
            key = self._cell_key(float(positions[i, 0]), float(positions[i, 1]))
            if key not in self._grid:
                self._grid[key] = []
            self._grid[key].append(i)

    def query_radius(self, x: float, y: float, radius: float, positions: np.ndarray) -> list[int]:
        """查找半径内的索引"""
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


@dataclass
class VectorizedNetwork:
    """向量化网络引擎 — 万级神经元加速

    状态存储为 NumPy 数组，批量计算 LIF 动力学。
    """
    # 神经元状态向量
    n: int = 0
    v: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    u: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    alive: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spike_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # 参数向量
    v_rest: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    v_reset: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    v_th: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    tau_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    r_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spike_cost: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    # 连接矩阵 (稀疏表示)
    weight_matrix: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    delay_matrix: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # 配置
    energy_decay: float = 0.005
    move_cost: float = 0.01
    input_current_scale: float = 1.0
    max_neurons: int = 1000
    scavenger_radius: float = 2.5
    reproduction_threshold: float = 2.5
    reproduction_share: float = 0.4
    mutation_scale: float = 0.05
    min_neurons: int = 2

    # 空间索引
    spatial: SpatialGrid = field(default_factory=SpatialGrid)

    # 统计
    total_spikes: int = 0
    total_ticks: int = 0
    birth_count: int = 0
    death_count: int = 0
    _pe_history: list[float] = field(default_factory=list)
    expected_state: np.ndarray | None = None
    schema_version: str = SCHEMA_VERSION

    def init_neurons(
        self,
        count: int,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_th: float = -50.0,
        tau_m: float = 20.0,
        r_m: float = 1.0,
        spike_cost_val: float = 0.05,
        energy_init: float = 1.0,
        spread: float = 10.0,
    ) -> None:
        """初始化 N 个 LIF 神经元"""
        self.n = count
        self.v = np.full(count, v_rest, dtype=np.float64)
        self.u = np.zeros(count, dtype=np.float64)
        self.energy = np.full(count, energy_init, dtype=np.float64)
        self.alive = np.ones(count, dtype=np.bool_)
        self.pos = np.random.uniform(-spread, spread, (count, 2))
        self.spike_count = np.zeros(count, dtype=np.int64)

        self.v_rest = np.full(count, v_rest, dtype=np.float64)
        self.v_reset = np.full(count, v_reset, dtype=np.float64)
        self.v_th = np.full(count, v_th, dtype=np.float64)
        self.tau_m = np.full(count, tau_m, dtype=np.float64)
        self.r_m = np.full(count, r_m, dtype=np.float64)
        self.spike_cost = np.full(count, spike_cost_val, dtype=np.float64)

        self.weight_matrix = np.zeros((count, count), dtype=np.float64)
        self.delay_matrix = np.zeros((count, count), dtype=np.int32)

    def connect_radius(self, radius: float, weight: float = 1.0, delay: int = 1) -> None:
        """基于距离自动连接"""
        for i in range(self.n):
            if not self.alive[i]:
                continue
            for j in range(self.n):
                if i == j or not self.alive[j]:
                    continue
                dx = self.pos[i, 0] - self.pos[j, 0]
                dy = self.pos[i, 1] - self.pos[j, 1]
                if dx * dx + dy * dy <= radius * radius:
                    self.weight_matrix[i, j] = weight
                    self.delay_matrix[i, j] = delay

    def connect_knn(self, k: int = 3, weight: float = 1.0, delay: int = 1) -> None:
        """k 近邻连接"""
        from numpy.linalg import norm
        for i in range(self.n):
            if not self.alive[i]:
                continue
            dists = norm(self.pos - self.pos[i], axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:k]
            for j in nearest:
                if self.alive[j]:
                    self.weight_matrix[i, j] = weight
                    self.delay_matrix[i, j] = delay

    def tick(self, tick: int, dt: float = 1.0, input_currents: np.ndarray | None = None) -> dict[str, Any]:
        """向量化 tick — 批量 LIF 更新"""
        if input_currents is None:
            input_currents = np.zeros(self.n, dtype=np.float64)

        alive_mask = self.alive

        # LIF 动力学 (向量化)
        dv = np.where(
            alive_mask,
            (-(self.v - self.v_rest) + self.r_m * input_currents * self.input_current_scale) / self.tau_m,
            0.0,
        )
        self.v = np.where(alive_mask, self.v + dv * dt, self.v)

        # NaN/Inf 保护
        nan_mask = ~np.isfinite(self.v)
        self.v[nan_mask] = self.v_reset[nan_mask]

        # 放电检测
        spike_mask = alive_mask & (self.v >= self.v_th)
        spike_indices = np.where(spike_mask)[0]
        tick_spikes = len(spike_indices)

        # 放电后重置
        self.v[spike_mask] = self.v_reset[spike_mask]
        self.energy[spike_mask] = np.maximum(0.0, self.energy[spike_mask] - self.spike_cost[spike_mask])
        self.spike_count[spike_mask] += 1

        # 传播: 放电神经元向连接目标注入电流 (下一 tick 生效)
        propagation_count = 0
        if tick_spikes > 0:
            for i in spike_indices:
                targets = np.where(self.weight_matrix[i] > 0)[0]
                propagation_count += len(targets)

        # 移动 (向量化)
        move_dx = np.random.uniform(-1.0, 1.0, self.n)
        move_dy = np.random.uniform(-1.0, 1.0, self.n)
        self.pos[alive_mask, 0] += move_dx[alive_mask]
        self.pos[alive_mask, 1] += move_dy[alive_mask]
        self.energy[alive_mask] = np.maximum(0.0, self.energy[alive_mask] - self.move_cost)

        # 能量衰减
        self.energy[alive_mask] = np.maximum(0.0, self.energy[alive_mask] - self.energy_decay)

        # 死亡判定
        death_mask = alive_mask & (self.energy <= 0.0)
        # 保护 min_neurons
        alive_after = int(np.sum(alive_mask)) - int(np.sum(death_mask))
        if alive_after < self.min_neurons:
            death_indices = np.where(death_mask)[0]
            if len(death_indices) > 0:
                save_count = self.min_neurons - alive_after
                sorted_indices = death_indices[np.argsort(-self.energy[death_indices])]
                saved = sorted_indices[:save_count]
                death_mask[saved] = False
                self.energy[saved] = 0.1

        deaths = int(np.sum(death_mask))
        self.alive[death_mask] = False
        self.death_count += deaths

        # 简单繁殖 (向量化 fitness 排序)
        births = 0
        alive_count = int(np.sum(self.alive))
        if alive_count < self.max_neurons:
            repro_mask = self.alive & (self.energy >= self.reproduction_threshold)
            repro_indices = np.where(repro_mask)[0]
            for i in repro_indices:
                if alive_count + births >= self.max_neurons:
                    break
                # 找到空位或跳过
                dead_slots = np.where(~self.alive)[0]
                if len(dead_slots) == 0:
                    break
                slot = dead_slots[0]
                child_energy = self.energy[i] * self.reproduction_share
                self.energy[i] -= child_energy

                self.v[slot] = self.v_rest[i]
                self.u[slot] = 0.0
                self.energy[slot] = child_energy
                self.alive[slot] = True
                self.pos[slot] = self.pos[i] + np.random.uniform(-0.5, 0.5, 2)
                self.spike_count[slot] = 0
                # 变异参数
                self.tau_m[slot] = self.tau_m[i] * (1.0 + np.random.uniform(-self.mutation_scale, self.mutation_scale))
                self.v_th[slot] = self.v_th[i] * (1.0 + np.random.uniform(-self.mutation_scale, self.mutation_scale))
                # 建立连接
                self.weight_matrix[i, slot] = 1.0
                self.weight_matrix[slot, i] = 1.0
                self.delay_matrix[i, slot] = 1
                self.delay_matrix[slot, i] = 1
                births += 1

        self.birth_count += births
        self.total_spikes += tick_spikes
        self.total_ticks += 1

        # 更新空间索引
        if self.total_ticks % 10 == 0:
            self.spatial.build(self.pos)

        # PE 计算
        alive_energies = self.energy[self.alive]
        mean_energy = float(np.mean(alive_energies)) if len(alive_energies) > 0 else 0.0
        actual = np.array([mean_energy])
        if self.expected_state is None:
            self.expected_state = actual.copy()
            pe = 0.0
        else:
            pe = float(np.sqrt(np.sum((actual - self.expected_state) ** 2)))
            self.expected_state = self.expected_state + 0.3 * (actual - self.expected_state)
        self._pe_history.append(pe)

        return {
            "schema_version": self.schema_version,
            "tick": tick,
            "alive_neurons": int(np.sum(self.alive)),
            "total_neurons": self.n,
            "tick_spikes": tick_spikes,
            "propagation_count": propagation_count,
            "births": births,
            "deaths": deaths,
            "prediction_error": pe,
            "mean_energy": mean_energy,
        }

    def get_spike_input_currents(self) -> np.ndarray:
        """根据上一次放电计算传播电流"""
        spike_mask = self.spike_count > 0  # 简化: 当前只用 spike_count > 0
        currents = np.zeros(self.n, dtype=np.float64)
        for i in np.where(spike_mask)[0]:
            targets = np.where(self.weight_matrix[i] > 0)[0]
            currents[targets] += self.weight_matrix[i, targets]
        return currents

    def get_summary(self) -> dict[str, Any]:
        alive_mask = self.alive
        alive_count = int(np.sum(alive_mask))
        return {
            "schema_version": self.schema_version,
            "total_neurons": self.n,
            "alive_neurons": alive_count,
            "total_spikes": self.total_spikes,
            "birth_count": self.birth_count,
            "death_count": self.death_count,
            "mean_energy": float(np.mean(self.energy[alive_mask])) if alive_count > 0 else 0.0,
            "pe_moving_avg": float(np.mean(self._pe_history[-10:])) if self._pe_history else 0.0,
        }
