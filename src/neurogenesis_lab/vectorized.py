"""向量化仿真引擎 — NumPy 加速，适用于万级神经元

接口兼容原有 Network，但内部状态用 NumPy 数组存储，
批量计算 LIF 动力学、空间索引、能量衰减等。
稀疏矩阵 (scipy.sparse) 存储连接权重，向量化放电传播。
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from neurogenesis_lab.config import (
    DECAY_HIGH_MULTIPLIER,
    DECAY_LOW_MULTIPLIER,
    DENSITY_HIGH_THRESHOLD,
    DENSITY_LOW_THRESHOLD,
    ENDANGERED_DECAY_DISCOUNT,
    ENDANGERED_EXIT_MULTIPLIER,
    MOVE_JITTER_RANGE,
    POPULATION_HIGH_THRESHOLD,
    POPULATION_LOW_THRESHOLD,
    REPRO_SHARE_HIGH_MULTIPLIER,
    REPRO_SHARE_LOW_MULTIPLIER,
    REPRO_SHARE_MAX,
    REPRO_SHARE_MIN,
    RESCUED_NEURON_ENERGY,
    SCHEMA_VERSION,
    SPAWN_POSITION_JITTER,
    PE_SMOOTHING_ALPHA,
    SPATIAL_REBUILD_INTERVAL,
    SPAWN_LINK_WEIGHT,
    SPAWN_LINK_DELAY,
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
    energy_cap: float = 5.0
    corpse_ttl: int = 50
    adaptive_decay: bool = True
    adaptive_reproduction: bool = True
    endangered_threshold: int = 10

    # 多模型类型向量（0=LIF, 1=Izhikevich, 2=AdEx）
    model_type: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    # Izhikevich 额外参数
    izh_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    izh_b: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    izh_c: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    izh_d: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # AdEx 额外参数
    adex_delta_t: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    adex_tau_w: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    adex_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    adex_b_param: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    adex_v_peak: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    # 适应电流 w（AdEx/Izh 恢复变量统一使用 self.u）

    # 尸体回收系统
    corpse_mask: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    corpse_timer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )
    corpse_energy: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    _corpse_decay_rate: float = 0.02

    # 濒危模式标志
    _endangered_mode: bool = False

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

    def _resolve_neuron_index(self, src_id: int | str) -> int | None:
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
        """初始化 N 个神经元，分配状态向量和空稀疏矩阵。

        支持混合模型初始化：默认全部 LIF，可通过 model_types 指定每个神经元的类型
        （0=LIF, 1=Izhikevich, 2=AdEx）。

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

        # 默认全部 LIF 类型
        self.model_type = np.zeros(count, dtype=np.int32)

        # Izhikevich 默认参数
        self.izh_a = np.full(count, 0.02, dtype=np.float64)
        self.izh_b = np.full(count, 0.2, dtype=np.float64)
        self.izh_c = np.full(count, -65.0, dtype=np.float64)
        self.izh_d = np.full(count, 8.0, dtype=np.float64)

        # AdEx 默认参数
        self.adex_delta_t = np.full(count, 2.0, dtype=np.float64)
        self.adex_tau_w = np.full(count, 30.0, dtype=np.float64)
        self.adex_a = np.full(count, 0.001, dtype=np.float64)
        self.adex_b_param = np.full(count, 0.0805, dtype=np.float64)
        self.adex_v_peak = np.full(count, 20.0, dtype=np.float64)

        # 尸体回收系统初始化
        self.corpse_mask = np.zeros(count, dtype=np.bool_)
        self.corpse_timer = np.zeros(count, dtype=np.int32)
        self.corpse_energy = np.zeros(count, dtype=np.float64)

        self.weight_matrix = csr_matrix((count, count), dtype=np.float64)
        self.delay_matrix = csr_matrix((count, count), dtype=np.int32)

    def connect_radius(
        self, radius: float, weight: float = 1.0, delay: int = 1
    ) -> None:
        """基于欧氏距离自动连接所有半径内的神经元对。

        使用 cKDTree 加速近邻搜索，复杂度 O(N log N)。

        Args:
            radius: 连接搜索半径。
            weight: 统一连接权重，默认 1.0。
            delay: 统一传播延迟（tick 数），默认 1。
        """
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            return
        tree = cKDTree(self.pos[alive_idx])
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        if len(pairs) == 0:
            self.weight_matrix = csr_matrix((self.n, self.n), dtype=np.float64)
            self.delay_matrix = csr_matrix((self.n, self.n), dtype=np.int32)
            return
        # pairs 是局部索引，映射回全局索引
        gi = alive_idx[pairs[:, 0]]
        gj = alive_idx[pairs[:, 1]]
        # 双向连接
        rows = np.concatenate([gi, gj])
        cols = np.concatenate([gj, gi])
        n_edges = len(rows)
        weights = np.full(n_edges, weight, dtype=np.float64)
        delays = np.full(n_edges, delay, dtype=np.int32)
        self.weight_matrix = csr_matrix((weights, (rows, cols)), shape=(self.n, self.n))
        self.delay_matrix = csr_matrix((delays, (rows, cols)), shape=(self.n, self.n))

    def connect_knn(self, k: int = 3, weight: float = 1.0, delay: int = 1) -> None:
        """k 近邻连接 — 每个存活神经元连接最近的 k 个邻居。

        使用 cKDTree 加速近邻搜索，复杂度 O(N log N)。

        Args:
            k: 近邻数量，默认 3。
            weight: 统一连接权重，默认 1.0。
            delay: 统一传播延迟（tick 数），默认 1。
        """
        alive_idx = np.where(self.alive)[0]
        if len(alive_idx) == 0:
            return
        alive_pos = self.pos[alive_idx]
        actual_k = min(k + 1, len(alive_idx))  # +1 因为包含自身
        tree = cKDTree(alive_pos)
        _, local_indices = tree.query(alive_pos, k=actual_k)
        # local_indices 形状 (n_alive, actual_k)
        if local_indices.ndim == 1:
            local_indices = local_indices[:, np.newaxis]
        n_alive = len(alive_idx)
        # 构建行列索引，排除自身（第 0 列是自身）
        src = np.repeat(alive_idx, actual_k - 1)
        dst = alive_idx[local_indices[:, 1:].ravel()]
        n_edges = len(src)
        weights = np.full(n_edges, weight, dtype=np.float64)
        delays = np.full(n_edges, delay, dtype=np.int32)
        self.weight_matrix = csr_matrix((weights, (src, dst)), shape=(self.n, self.n))
        self.delay_matrix = csr_matrix((delays, (src, dst)), shape=(self.n, self.n))

    def tick(
        self, tick: int, dt: float = 1.0, input_currents: np.ndarray | None = None
    ) -> TickSnapshot:
        """执行一个向量化仿真时间步 — 批量多模型更新 + 稀疏矩阵传播。

        支持 LIF / Izhikevich / AdEx 混合模型，通过 model_type 掩码分组计算。

        Args:
            tick: 当前时间步编号。
            dt: 时间步长，单位 ms，默认 1.0。
            input_currents: 外部输入电流向量，形状 (n,)，None 时为零输入。

        Returns:
            本 tick 的快照字典。

        Raises:
            ValueError: input_currents 形状与神经元数量不匹配时抛出。
        """
        # 濒危模式判定
        alive_count = int(np.sum(self.alive))
        if alive_count <= self.endangered_threshold and not self._endangered_mode:
            self._endangered_mode = True
        if alive_count > self.endangered_threshold * ENDANGERED_EXIT_MULTIPLIER:
            self._endangered_mode = False

        combined_inputs, ready_events = self._inject_inputs(tick, dt, input_currents)
        alive_mask, spike_indices, tick_spikes = self._vectorized_dynamics_step(
            dt, combined_inputs
        )
        propagation_count = self._propagate_spikes(tick, spike_indices)
        self._update_corpses()
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

    # ------------------------------------------------------------------
    # 多模型动力学步进
    # ------------------------------------------------------------------

    def _vectorized_dynamics_step(
        self, dt: float, combined_inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """统一多模型动力学调度器 — 按 model_type 分组计算。

        model_type: 0=LIF, 1=Izhikevich, 2=AdEx。

        Args:
            dt: 时间步长 (ms)。
            combined_inputs: 汇总输入电流向量 (n,)。

        Returns:
            (alive_mask, spike_indices, tick_spikes)
        """
        alive_mask = self.alive.copy()
        scaled_input = combined_inputs * self.input_current_scale

        # LIF 子集 (model_type == 0)
        lif_mask = alive_mask & (self.model_type == 0)
        if np.any(lif_mask):
            self._lif_step(dt, scaled_input, lif_mask)

        # Izhikevich 子集 (model_type == 1)
        izh_mask = alive_mask & (self.model_type == 1)
        if np.any(izh_mask):
            self._izh_step(dt, scaled_input, izh_mask)

        # AdEx 子集 (model_type == 2)
        adex_mask = alive_mask & (self.model_type == 2)
        if np.any(adex_mask):
            self._adex_step(dt, scaled_input, adex_mask)

        # NaN / Inf 安全回退
        nan_mask = ~np.isfinite(self.v)
        self.v[nan_mask] = self.v_reset[nan_mask]

        # 收集放电
        spike_mask = alive_mask & (self.v >= self.v_th)
        spike_indices = np.where(spike_mask)[0]
        tick_spikes = len(spike_indices)

        # 重置放电神经元
        # LIF / Izh 重置
        lif_izh_spike = spike_mask & (self.model_type != 2)
        self.v[lif_izh_spike] = self.v_reset[lif_izh_spike]
        # Izh 恢复变量重置
        izh_spike = spike_mask & (self.model_type == 1)
        self.v[izh_spike] = self.izh_c[izh_spike]
        self.u[izh_spike] += self.izh_d[izh_spike]
        # AdEx 重置
        adex_spike = spike_mask & (self.model_type == 2)
        self.v[adex_spike] = self.v_reset[adex_spike]
        self.u[adex_spike] += self.adex_b_param[adex_spike]

        # 能量消耗 & 放电计数
        self.energy[spike_mask] = np.maximum(
            0.0, self.energy[spike_mask] - self.spike_cost[spike_mask]
        )
        self.spike_count[spike_mask] += 1
        self._last_spike_mask = spike_mask.copy()
        return alive_mask, spike_indices, tick_spikes

    def _lif_step(self, dt: float, scaled_input: np.ndarray, mask: np.ndarray) -> None:
        """LIF (Leaky Integrate-and-Fire) 向量化步进。"""
        dv = (
            -(self.v[mask] - self.v_rest[mask]) + self.r_m[mask] * scaled_input[mask]
        ) / self.tau_m[mask]
        self.v[mask] += dv * dt

    def _izh_step(self, dt: float, scaled_input: np.ndarray, mask: np.ndarray) -> None:
        """Izhikevich 模型向量化步进。

        dv/dt = 0.04v² + 5v + 140 - u + I
        du/dt = a(bv - u)
        """
        v = self.v[mask]
        u = self.u[mask]
        I = scaled_input[mask]  # noqa: E741
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I
        du = self.izh_a[mask] * (self.izh_b[mask] * v - u)
        self.v[mask] += dv * dt
        self.u[mask] += du * dt

    def _adex_step(self, dt: float, scaled_input: np.ndarray, mask: np.ndarray) -> None:
        """AdEx (Adaptive Exponential) 模型向量化步进。

        C dv/dt = -g_L(v - E_L) + g_L Δ_T exp((v - V_T)/Δ_T) - w + I
        τ_w dw/dt = a(v - E_L) - w

        简化：使用 tau_m 替代 C/g_L，v_rest 替代 E_L，v_th 替代 V_T。
        """
        v = self.v[mask]
        u = self.u[mask]
        I = scaled_input[mask]  # noqa: E741
        delta_t = self.adex_delta_t[mask]
        # 指数项，裁切避免溢出
        exp_arg = np.clip(
            (v - self.v_th[mask]) / np.maximum(delta_t, 1e-6), -50.0, 50.0
        )
        dv = (
            -(v - self.v_rest[mask])
            + delta_t * np.exp(exp_arg)
            - self.r_m[mask] * u
            + self.r_m[mask] * I
        ) / self.tau_m[mask]
        du = (self.adex_a[mask] * (v - self.v_rest[mask]) - u) / self.adex_tau_w[mask]
        self.v[mask] += dv * dt
        self.u[mask] += du * dt

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

    # ------------------------------------------------------------------
    # 尸体回收系统
    # ------------------------------------------------------------------

    def _update_corpses(self) -> None:
        """更新尸体状态 — TTL 递减、能量衰减、附近神经元拾取。"""
        if not np.any(self.corpse_mask):
            return

        # TTL 递减
        self.corpse_timer[self.corpse_mask] -= 1
        # 能量衰减
        self.corpse_energy[self.corpse_mask] = np.maximum(
            0.0, self.corpse_energy[self.corpse_mask] - self._corpse_decay_rate
        )

        # 拾荒：存活神经元吸收半径内尸体能量
        corpse_indices = np.where(self.corpse_mask & (self.corpse_energy > 0))[0]
        for ci in corpse_indices:
            cx, cy = float(self.pos[ci, 0]), float(self.pos[ci, 1])
            nearby = self.spatial.query_radius(cx, cy, self.scavenger_radius, self.pos)
            # 过滤出存活且非尸体的神经元
            alive_nearby = [idx for idx in nearby if idx != ci and self.alive[idx]]
            if alive_nearby:
                # 选择最近的一个
                dists = [
                    (self.pos[idx, 0] - cx) ** 2 + (self.pos[idx, 1] - cy) ** 2
                    for idx in alive_nearby
                ]
                nearest = alive_nearby[int(np.argmin(dists))]
                self.energy[nearest] += self.corpse_energy[ci]
                self.corpse_energy[ci] = 0.0

        # 清理过期或无能量尸体
        expired = self.corpse_mask & (
            (self.corpse_timer <= 0) | (self.corpse_energy <= 0)
        )
        self.corpse_mask[expired] = False
        self.corpse_timer[expired] = 0
        self.corpse_energy[expired] = 0.0

    # ------------------------------------------------------------------
    # 自适应衰减 & 繁殖
    # ------------------------------------------------------------------

    def _calculate_adaptive_decay(self, alive_count: int) -> float:
        """根据种群密度计算自适应能量衰减系数。

        Args:
            alive_count: 当前存活神经元数量。

        Returns:
            调整后的能量衰减值。
        """
        if not self.adaptive_decay:
            return self.energy_decay
        density_ratio = alive_count / self.max_neurons if self.max_neurons > 0 else 0.5
        if density_ratio > DENSITY_HIGH_THRESHOLD:
            return self.energy_decay * DECAY_HIGH_MULTIPLIER
        elif density_ratio < DENSITY_LOW_THRESHOLD:
            return self.energy_decay * DECAY_LOW_MULTIPLIER
        return self.energy_decay

    def _calculate_adaptive_reproduction_share(self, alive_count: int) -> float:
        """根据种群数量计算自适应繁殖份额。

        Args:
            alive_count: 当前存活神经元数量。

        Returns:
            调整后的繁殖能量分享比例。
        """
        if not self.adaptive_reproduction:
            return self.reproduction_share
        pop_ratio = alive_count / self.max_neurons if self.max_neurons > 0 else 0.5
        if pop_ratio > POPULATION_HIGH_THRESHOLD:
            return min(
                REPRO_SHARE_MAX,
                self.reproduction_share * REPRO_SHARE_HIGH_MULTIPLIER,
            )
        elif pop_ratio < POPULATION_LOW_THRESHOLD:
            return max(
                REPRO_SHARE_MIN,
                self.reproduction_share * REPRO_SHARE_LOW_MULTIPLIER,
            )
        return self.reproduction_share

    # ------------------------------------------------------------------
    # 演化 tick
    # ------------------------------------------------------------------

    def _run_vectorized_evolution(
        self, tick: int, alive_mask: np.ndarray
    ) -> tuple[int, int]:
        _ = tick
        alive_count = int(np.sum(self.alive))

        # 移动
        move_dx = np.random.uniform(-MOVE_JITTER_RANGE, MOVE_JITTER_RANGE, self.n)
        move_dy = np.random.uniform(-MOVE_JITTER_RANGE, MOVE_JITTER_RANGE, self.n)
        self.pos[alive_mask, 0] += move_dx[alive_mask]
        self.pos[alive_mask, 1] += move_dy[alive_mask]
        self.energy[alive_mask] = np.maximum(
            0.0, self.energy[alive_mask] - self.move_cost
        )

        # 自适应衰减（含濒危折扣）
        effective_decay = self._calculate_adaptive_decay(alive_count)
        if self._endangered_mode:
            effective_decay *= ENDANGERED_DECAY_DISCOUNT
        self.energy[alive_mask] = np.maximum(
            0.0, self.energy[alive_mask] - effective_decay
        )

        # 能量上限裁切
        self.energy[alive_mask] = np.minimum(self.energy_cap, self.energy[alive_mask])

        # 死亡判定
        death_mask = alive_mask & (self.energy <= 0.0)
        alive_after = alive_count - int(np.sum(death_mask))
        if alive_after < self.min_neurons:
            death_indices = np.where(death_mask)[0]
            if len(death_indices) > 0:
                save_count = self.min_neurons - alive_after
                sorted_indices = death_indices[np.argsort(-self.energy[death_indices])]
                saved = sorted_indices[:save_count]
                death_mask[saved] = False
                self.energy[saved] = RESCUED_NEURON_ENERGY

        deaths = int(np.sum(death_mask))
        # 死亡神经元 → 尸体
        dead_indices = np.where(death_mask)[0]
        for di in dead_indices:
            self.corpse_mask[di] = True
            self.corpse_timer[di] = self.corpse_ttl
            self.corpse_energy[di] = max(0.0, float(self.energy[di]))
        self.alive[death_mask] = False
        self.death_count += deaths

        # 繁殖
        births = 0
        alive_count = int(np.sum(self.alive))
        effective_share = self._calculate_adaptive_reproduction_share(alive_count)
        if alive_count < self.max_neurons:
            repro_mask = self.alive & (self.energy >= self.reproduction_threshold)
            repro_indices = np.where(repro_mask)[0]
            if len(repro_indices) > 0:
                new_rows: list[int] = []
                new_cols: list[int] = []
                new_weights: list[float] = []
                new_delays: list[int] = []
                for i in repro_indices:
                    if alive_count + births >= self.max_neurons:
                        break
                    dead_slots = np.where(~self.alive & ~self.corpse_mask)[0]
                    if len(dead_slots) == 0:
                        break
                    slot = int(dead_slots[0])
                    child_energy = self.energy[i] * effective_share
                    self.energy[i] -= child_energy

                    self.v[slot] = self.v_rest[i]
                    self.u[slot] = 0.0
                    self.energy[slot] = child_energy
                    self.alive[slot] = True
                    self.corpse_mask[slot] = False
                    self.corpse_timer[slot] = 0
                    self.corpse_energy[slot] = 0.0
                    self.pos[slot] = self.pos[i] + np.random.uniform(
                        -SPAWN_POSITION_JITTER, SPAWN_POSITION_JITTER, 2
                    )
                    self.spike_count[slot] = 0
                    # 继承模型类型
                    self.model_type[slot] = self.model_type[i]
                    self.tau_m[slot] = self.tau_m[i] * (
                        1.0
                        + np.random.uniform(-self.mutation_scale, self.mutation_scale)
                    )
                    self.v_th[slot] = self.v_th[i] * (
                        1.0
                        + np.random.uniform(-self.mutation_scale, self.mutation_scale)
                    )
                    new_rows.extend([int(i), slot])
                    new_cols.extend([slot, int(i)])
                    new_weights.extend([SPAWN_LINK_WEIGHT, SPAWN_LINK_WEIGHT])
                    new_delays.extend([SPAWN_LINK_DELAY, SPAWN_LINK_DELAY])
                    births += 1
                if new_rows:
                    shape = (self.n, self.n)
                    delta_w = coo_matrix(
                        (new_weights, (new_rows, new_cols)), shape=shape
                    ).tocsr()
                    delta_d = coo_matrix(
                        (new_delays, (new_rows, new_cols)), shape=shape, dtype=np.int32
                    ).tocsr()
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
        if self.total_ticks % SPATIAL_REBUILD_INTERVAL == 0:
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
            self.expected_state = self.expected_state + PE_SMOOTHING_ALPHA * (
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

    # ------------------------------------------------------------------
    # Checkpoint 持久化
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, save_tick: int = 0) -> None:
        """将向量化网络完整状态序列化为 pickle 文件。

        Args:
            path: checkpoint 文件路径。
            save_tick: 当前 tick 编号，记入 payload。
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "tick": save_tick,
            "random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "network": self,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> tuple["VectorizedNetwork", int]:
        """从 pickle checkpoint 恢复向量化网络实例。

        Args:
            path: checkpoint 文件路径。

        Returns:
            (network, tick) 恢复的网络实例和保存时的 tick 编号。

        Raises:
            ValueError: schema 版本不兼容时抛出。
            TypeError: 序列化对象类型不匹配时抛出。
        """
        filepath = Path(path)
        with open(filepath, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        saved_version = payload.get("schema_version", "0.0")
        current_major = SCHEMA_VERSION.split(".")[0]
        saved_major = saved_version.split(".")[0]
        if current_major != saved_major:
            raise ValueError(
                f"Schema 版本不兼容：当前 {SCHEMA_VERSION}，checkpoint {saved_version}"
            )
        net = payload.get("network")
        if not isinstance(net, cls):
            raise TypeError(
                f"Checkpoint 中的对象类型不匹配：期望 {cls.__name__}，"
                f"实际 {type(net).__name__}"
            )
        tick = payload.get("tick", 0)
        random.setstate(payload["random_state"])
        if "numpy_random_state" in payload:
            np.random.set_state(payload["numpy_random_state"])
        return net, tick
