"""向量化仿真引擎 — NumPy 加速，适用于万级神经元

接口兼容原有 Network，但内部状态用 NumPy 数组存储，
批量计算 LIF 动力学、空间索引、能量衰减等。
稀疏矩阵 (scipy.sparse) 存储连接权重，向量化放电传播。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from scipy.spatial.distance import cdist

from neurogenesis_lab.config import (
    MOVE_JITTER_RANGE,
    RESCUED_NEURON_ENERGY,
    SCHEMA_VERSION,
    SPAWN_POSITION_JITTER,
)
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import safe_float
from neurogenesis_lab.protocols import TickSnapshot, EngineSummary


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

    def query_radius(
        self, x: float, y: float, radius: float, positions: np.ndarray
    ) -> list[int]:
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


@dataclass
class VectorizedNetwork:
    """向量化网络引擎 — 万级神经元加速。

    状态存储为 NumPy 数组，批量计算 LIF 动力学。
    连接权重使用 scipy.sparse CSR 矩阵，向量化放电传播。
    实现 SimEngine 协议，返回统一的 TickSnapshot 和 EngineSummary。

    Attributes:
        n: 神经元总数（含已死亡槽位）。
        v: 膜电位向量，形状 (n,)，单位 mV。
        u: 恢复变量向量，形状 (n,)。
        energy: 能量向量，形状 (n,)。
        alive: 存活掩码向量，形状 (n,)，dtype bool。
        pos: 位置矩阵，形状 (n, 2)。
        spike_count: 累计放电次数向量，形状 (n,)。
        weight_matrix: 连接权重稀疏矩阵 (n, n)。
        delay_matrix: 传播延迟稀疏矩阵 (n, n)。
        energy_decay: 每 tick 基础能量衰减量，默认 0.005。
        schema_version: 数据格式版本号。

    Example:
        >>> vnet = VectorizedNetwork()
        >>> vnet.init_neurons(1000, spread=20.0)
        >>> vnet.connect_knn(k=5)
        >>> snapshot = vnet.tick(tick=1)
    """

    # 神经元状态向量
    n: int = 0
    v: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    u: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    energy: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    alive: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    pos: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spike_count: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )

    # 参数向量
    v_rest: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    v_reset: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    v_th: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    tau_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    r_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    spike_cost: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # 连接矩阵 (scipy 稀疏)
    weight_matrix: csr_matrix | None = None
    delay_matrix: csr_matrix | None = None

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
    # 事件调度（tick -> 输入电流向量）
    scheduled_inputs: dict[int, np.ndarray] = field(default_factory=dict)
    _last_spike_mask: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )

    # 统计
    total_spikes: int = 0
    total_propagations: int = 0
    total_ticks: int = 0
    birth_count: int = 0
    death_count: int = 0
    _pe_history: list[float] = field(default_factory=list)
    expected_state: np.ndarray | None = None
    schema_version: str = SCHEMA_VERSION

    def _resolve_neuron_index(self, src_id: Any) -> int | None:
        if isinstance(src_id, int):
            idx = src_id
        elif isinstance(src_id, str):
            text = src_id.strip()
            if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                idx = int(text)
            else:
                return None
        else:
            return None
        if 0 <= idx < self.n:
            return idx
        return None

    def _schedule_currents(self, tick: int, currents: np.ndarray) -> None:
        if tick in self.scheduled_inputs:
            self.scheduled_inputs[tick] += currents
        else:
            self.scheduled_inputs[tick] = currents.copy()

    def enqueue(self, events: list[Event]) -> None:
        """注入外部事件，按 tick 调度到目标神经元输入电流。

        Args:
            events: 待注入的事件列表，event.src_id 解析为目标神经元索引。
        """
        for event in events:
            target_idx = self._resolve_neuron_index(event.src_id)
            if target_idx is None:
                continue
            scheduled_tick = max(0, int(event.t))
            weight = float(event.meta.get("weight", 1.0))
            if not np.isfinite(weight):
                continue
            payload = np.zeros(self.n, dtype=np.float64)
            payload[target_idx] = weight
            self._schedule_currents(scheduled_tick, payload)

    def init_neurons(
        self,
        count: int,
        v_rest: float = -65.0,
        v_reset: float = -65.0,
        v_th: float = -55.0,  # 与 NeuronParams 默认值对齐（原值 -50.0）
        tau_m: float = 20.0,
        r_m: float = 1.0,
        spike_cost_val: float = 0.05,
        energy_init: float = 1.0,
        spread: float = 10.0,
    ) -> None:
        """初始化 N 个 LIF 神经元，分配状态向量和空稀疏矩阵。

        Args:
            count: 神经元数量。
            v_rest: 静息电位，单位 mV，默认 -65.0。
            v_reset: 重置电位，单位 mV，默认 -65.0。
            v_th: 放电阈值，单位 mV，默认 -55.0。
            tau_m: 膜时间常数，单位 ms，默认 20.0。
            r_m: 膜电阻缩放因子，默认 1.0。
            spike_cost_val: 每次放电能量消耗，默认 0.05。
            energy_init: 初始能量，默认 1.0。
            spread: 位置随机分布范围 [-spread, spread]，默认 10.0。
        """
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

        self.weight_matrix = csr_matrix((count, count), dtype=np.float64)
        self.delay_matrix = csr_matrix((count, count), dtype=np.int32)

    def connect_radius(
        self, radius: float, weight: float = 1.0, delay: int = 1
    ) -> None:
        """基于欧氏距离自动连接所有半径内的神经元对。

        Args:
            radius: 连接搜索半径。
            weight: 统一连接权重，默认 1.0。
            delay: 统一传播延迟（tick 数），默认 1。
        """
        dists = cdist(self.pos, self.pos)
        mask = (dists <= radius) & (dists > 0) & np.outer(self.alive, self.alive)
        lil_w = lil_matrix((self.n, self.n), dtype=np.float64)
        lil_d = lil_matrix((self.n, self.n), dtype=np.int32)
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            lil_w[r, c] = weight
            lil_d[r, c] = delay
        self.weight_matrix = csr_matrix(lil_w)
        self.delay_matrix = csr_matrix(lil_d)

    def connect_knn(self, k: int = 3, weight: float = 1.0, delay: int = 1) -> None:
        """k 近邻连接 — 每个存活神经元连接最近的 k 个邻居。

        Args:
            k: 近邻数量，默认 3。
            weight: 统一连接权重，默认 1.0。
            delay: 统一传播延迟（tick 数），默认 1。
        """
        from numpy.linalg import norm

        lil_w = lil_matrix((self.n, self.n), dtype=np.float64)
        lil_d = lil_matrix((self.n, self.n), dtype=np.int32)
        for i in range(self.n):
            if not self.alive[i]:
                continue
            dists = norm(self.pos - self.pos[i], axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:k]
            for j in nearest:
                if self.alive[j]:
                    lil_w[i, j] = weight
                    lil_d[i, j] = delay
        self.weight_matrix = csr_matrix(lil_w)
        self.delay_matrix = csr_matrix(lil_d)

    def tick(
        self, tick: int, dt: float = 1.0, input_currents: np.ndarray | None = None
    ) -> TickSnapshot:
        """执行一个向量化仿真时间步 — 批量 LIF 更新 + 稀疏矩阵传播。

        Args:
            tick: 当前时间步编号。
            dt: 时间步长，单位 ms，默认 1.0。
            input_currents: 外部输入电流向量，形状 (n,)，None 时为零输入。

        Returns:
            本 tick 的快照字典。

        Raises:
            ValueError: input_currents 形状与神经元数量不匹配时抛出。
        """
        combined_inputs, ready_events = self._inject_inputs(tick, dt, input_currents)
        alive_mask, spike_indices, tick_spikes = self._vectorized_lif_step(
            dt, combined_inputs
        )
        propagation_count = self._propagate_spikes(tick, spike_indices)
        births, deaths = self._run_vectorized_evolution(tick, alive_mask)
        self.total_spikes += tick_spikes
        self.total_ticks += 1
        self._cleanup_scheduled_inputs(int(tick))
        return self._collect_vectorized_stats(
            tick=tick,
            ready_events=ready_events,
            tick_spikes=tick_spikes,
            propagation_count=propagation_count,
            births=births,
            deaths=deaths,
        )

    def _inject_inputs(
        self, tick: int, dt: float, input_currents: np.ndarray | None
    ) -> tuple[np.ndarray, int]:
        _ = dt
        if input_currents is None:
            base_inputs = np.zeros(self.n, dtype=np.float64)
        else:
            base_inputs = np.asarray(input_currents, dtype=np.float64)
            if base_inputs.shape != (self.n,):
                raise ValueError(
                    f"input_currents 形状非法: 期望 ({self.n},), 实际 {base_inputs.shape}"
                )
        routed_inputs = np.zeros(self.n, dtype=np.float64)
        due_ticks = [k for k in self.scheduled_inputs if k <= int(tick)]
        for due_tick in due_ticks:
            routed_inputs += self.scheduled_inputs.pop(due_tick)
        ready_events = int(np.count_nonzero(routed_inputs))
        combined_inputs = base_inputs + routed_inputs
        if combined_inputs.size:
            combined_inputs = np.nan_to_num(
                combined_inputs, nan=0.0, posinf=0.0, neginf=0.0
            )
        return combined_inputs, ready_events

    def _vectorized_lif_step(
        self, dt: float, combined_inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        alive_mask = self.alive
        dv = np.where(
            alive_mask,
            (
                -(self.v - self.v_rest)
                + self.r_m * combined_inputs * self.input_current_scale
            )
            / self.tau_m,
            0.0,
        )
        self.v = np.where(alive_mask, self.v + dv * dt, self.v)

        nan_mask = ~np.isfinite(self.v)
        self.v[nan_mask] = self.v_reset[nan_mask]

        spike_mask = alive_mask & (self.v >= self.v_th)
        spike_indices = np.where(spike_mask)[0]
        tick_spikes = len(spike_indices)
        self.v[spike_mask] = self.v_reset[spike_mask]
        self.energy[spike_mask] = np.maximum(
            0.0, self.energy[spike_mask] - self.spike_cost[spike_mask]
        )
        self.spike_count[spike_mask] += 1
        self._last_spike_mask = spike_mask.copy()
        return alive_mask, spike_indices, tick_spikes

    def _propagate_spikes(self, tick: int, spike_indices: np.ndarray) -> int:
        propagation_count = 0
        if len(spike_indices) <= 0:
            return propagation_count
        for spike_idx in spike_indices:
            row_w = self.weight_matrix.getrow(spike_idx)
            if row_w.nnz <= 0:
                continue
            row_d = self.delay_matrix.getrow(spike_idx)
            delay_map = {
                int(col): int(delay)
                for col, delay in zip(row_d.indices.tolist(), row_d.data.tolist())
            }
            targets = row_w.indices
            weights = row_w.data
            propagation_count += len(targets)
            for target, weight in zip(targets, weights):
                if not self.alive[target]:
                    continue
                delay = max(0, delay_map.get(int(target), 1))
                deliver_tick = int(tick) + delay
                payload = np.zeros(self.n, dtype=np.float64)
                payload[int(target)] = float(weight)
                self._schedule_currents(deliver_tick, payload)
        self.total_propagations += propagation_count
        return propagation_count

    def _run_vectorized_evolution(
        self, tick: int, alive_mask: np.ndarray
    ) -> tuple[int, int]:
        _ = tick
        move_dx = np.random.uniform(-MOVE_JITTER_RANGE, MOVE_JITTER_RANGE, self.n)
        move_dy = np.random.uniform(-MOVE_JITTER_RANGE, MOVE_JITTER_RANGE, self.n)
        self.pos[alive_mask, 0] += move_dx[alive_mask]
        self.pos[alive_mask, 1] += move_dy[alive_mask]
        self.energy[alive_mask] = np.maximum(
            0.0, self.energy[alive_mask] - self.move_cost
        )
        self.energy[alive_mask] = np.maximum(
            0.0, self.energy[alive_mask] - self.energy_decay
        )

        death_mask = alive_mask & (self.energy <= 0.0)
        alive_after = int(np.sum(alive_mask)) - int(np.sum(death_mask))
        if alive_after < self.min_neurons:
            death_indices = np.where(death_mask)[0]
            if len(death_indices) > 0:
                save_count = self.min_neurons - alive_after
                sorted_indices = death_indices[np.argsort(-self.energy[death_indices])]
                saved = sorted_indices[:save_count]
                death_mask[saved] = False
                self.energy[saved] = RESCUED_NEURON_ENERGY

        deaths = int(np.sum(death_mask))
        self.alive[death_mask] = False
        self.death_count += deaths

        births = 0
        alive_count = int(np.sum(self.alive))
        if alive_count < self.max_neurons:
            repro_mask = self.alive & (self.energy >= self.reproduction_threshold)
            repro_indices = np.where(repro_mask)[0]
            if len(repro_indices) > 0:
                # 收集新边，最后用 COO 批量合并，避免 lil 中间格式开销
                new_rows: list[int] = []
                new_cols: list[int] = []
                new_weights: list[float] = []
                new_delays: list[int] = []
                for i in repro_indices:
                    if alive_count + births >= self.max_neurons:
                        break
                    dead_slots = np.where(~self.alive)[0]
                    if len(dead_slots) == 0:
                        break
                    slot = int(dead_slots[0])
                    child_energy = self.energy[i] * self.reproduction_share
                    self.energy[i] -= child_energy

                    self.v[slot] = self.v_rest[i]
                    self.u[slot] = 0.0
                    self.energy[slot] = child_energy
                    self.alive[slot] = True
                    self.pos[slot] = self.pos[i] + np.random.uniform(-SPAWN_POSITION_JITTER, SPAWN_POSITION_JITTER, 2)
                    self.spike_count[slot] = 0
                    self.tau_m[slot] = self.tau_m[i] * (
                        1.0
                        + np.random.uniform(-self.mutation_scale, self.mutation_scale)
                    )
                    self.v_th[slot] = self.v_th[i] * (
                        1.0
                        + np.random.uniform(-self.mutation_scale, self.mutation_scale)
                    )
                    # 双向连接：parent ↔ child
                    new_rows.extend([int(i), slot])
                    new_cols.extend([slot, int(i)])
                    new_weights.extend([1.0, 1.0])
                    new_delays.extend([1, 1])
                    births += 1
                if new_rows:
                    shape = (self.n, self.n)
                    delta_w = coo_matrix(
                        (new_weights, (new_rows, new_cols)), shape=shape
                    ).tocsr()
                    delta_d = coo_matrix(
                        (new_delays, (new_rows, new_cols)), shape=shape, dtype=np.int32
                    ).tocsr()
                    # 合并：新边覆盖旧边（取最大值即可，因为新边权重固定为 1）
                    self.weight_matrix = self.weight_matrix + delta_w
                    self.delay_matrix = self.delay_matrix + delta_d
        self.birth_count += births
        return births, deaths

    def _cleanup_scheduled_inputs(self, current_tick: int) -> None:
        stale_ticks = [k for k in self.scheduled_inputs if k < current_tick]
        for stale_tick in stale_ticks:
            self.scheduled_inputs.pop(stale_tick, None)

    def _collect_vectorized_stats(
        self,
        *,
        tick: int,
        ready_events: int,
        tick_spikes: int,
        propagation_count: int,
        births: int,
        deaths: int,
    ) -> TickSnapshot:

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
            self.expected_state = self.expected_state + 0.3 * (
                actual - self.expected_state
            )
        self._pe_history.append(pe)
        return {
            "schema_version": self.schema_version,
            "tick": tick,
            "ready_events": ready_events,
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
        """根据上一次放电掩码计算传播电流向量。

        利用稀疏矩阵行求和，将放电神经元的出边权重汇总到目标。

        Returns:
            形状为 (n,) 的电流向量。
        """
        spike_mask = self._last_spike_mask
        if spike_mask.size == 0:
            return np.zeros(self.n, dtype=np.float64)
        spike_indices = np.where(spike_mask)[0]
        if len(spike_indices) == 0:
            return np.zeros(self.n, dtype=np.float64)
        return np.asarray(self.weight_matrix[spike_indices].sum(axis=0)).ravel()

    def get_summary(self) -> EngineSummary:
        """生成向量化网络运行摘要。

        Returns:
            包含存活数、放电数、能量均值等统计信息的字典。
        """
        alive_mask = self.alive
        alive_count = int(np.sum(alive_mask))
        return {
            "schema_version": self.schema_version,
            "total_neurons": self.n,
            "alive_neurons": alive_count,
            "total_spikes": self.total_spikes,
            "total_propagations": self.total_propagations,
            "birth_count": self.birth_count,
            "death_count": self.death_count,
            "mean_energy": (
                float(np.mean(self.energy[alive_mask])) if alive_count > 0 else 0.0
            ),
            "pe_moving_avg": (
                float(np.mean(self._pe_history[-10:])) if self._pe_history else 0.0
            ),
            "active_ratio": alive_count / self.n if self.n > 0 else 0.0,
        }
