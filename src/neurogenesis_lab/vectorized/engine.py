"""向量化仿真引擎 — NumPy 加速，适用于万级神经元

接口兼容原有 Network，但内部状态用 NumPy 数组存储，
批量计算 LIF 动力学、空间索引、能量衰减等。
稀疏矩阵 (scipy.sparse) 存储连接权重，向量化放电传播。

``VectorizedNetwork`` 由多个功能 Mixin（dynamics / propagation / lifecycle /
persistence）组合而成，全部字段经多重继承从 ``VectorizedState`` 获得。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from neurogenesis_lab.config import (
    PE_SMOOTHING_ALPHA,
    SPATIAL_REBUILD_INTERVAL,
)
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.evolution import update_endangered_mode
from neurogenesis_lab.protocols import EngineSummary, TickSnapshot
from neurogenesis_lab.vectorized.dynamics import DynamicsMixin
from neurogenesis_lab.vectorized.lifecycle import LifecycleMixin
from neurogenesis_lab.vectorized.persistence import PersistenceMixin
from neurogenesis_lab.vectorized.propagation import PropagationMixin


@dataclass
class VectorizedNetwork(
    DynamicsMixin,
    PropagationMixin,
    LifecycleMixin,
    PersistenceMixin,
):
    """向量化网络引擎 — 万级神经元加速。

    状态存储为 NumPy 数组，批量计算 LIF 动力学。
    连接权重使用 scipy.sparse CSR 矩阵，向量化放电传播。
    实现 SimEngine 协议，返回统一的 TickSnapshot 和 EngineSummary。

    全部字段声明见 ``VectorizedState``；本类仅承载对外接口方法与 tick 编排。

    Example:
        >>> vnet = VectorizedNetwork()
        >>> vnet.init_neurons(1000, spread=20.0)
        >>> vnet.connect_knn(k=5)
        >>> snapshot = vnet.tick(tick=1)
    """

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
            self._write_token(target_idx, int(event.token_id))
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
        self.token_memory = np.zeros((count, 64), dtype=np.int32)
        self.token_mem_len = np.zeros(count, dtype=np.int32)
        self.token_mem_ptr = np.zeros(count, dtype=np.int32)
        self.prediction_hits = np.zeros(count, dtype=np.int64)

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

    def connect_radius(self, radius: float, weight: float = 1.0, delay: int = 1) -> None:
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
        self._endangered_mode = update_endangered_mode(
            self._endangered_mode, alive_count, self.endangered_threshold
        )

        combined_inputs, ready_events = self._inject_inputs(tick, dt, input_currents)
        alive_mask, spike_indices, tick_spikes = self._vectorized_dynamics_step(dt, combined_inputs)
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
            "mean_energy": (float(np.mean(self.energy[alive_mask])) if alive_count > 0 else 0.0),
            "pe_moving_avg": (float(np.mean(self._pe_history[-10:])) if self._pe_history else 0.0),
            "active_ratio": alive_count / self.n if self.n > 0 else 0.0,
        }
