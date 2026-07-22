"""向量化放电传播 — vectorized 子包的稀疏矩阵传播部分。"""

from __future__ import annotations

import numpy as np

from neurogenesis_lab.vectorized.state import VectorizedState


class PropagationMixin(VectorizedState):
    """基于稀疏矩阵切片 + 按延迟分组的批量放电传播。"""

    def _propagate_spikes(self, tick: int, spike_indices: np.ndarray) -> int:
        """批量向量化放电传播 — 使用稀疏矩阵切片 + 按延迟分组调度。

        相比逐神经元 getrow + 逐目标 payload 分配，本实现：
        1. 一次性取出所有放电源的出边（稀疏矩阵切片）
        2. 转换为 COO 格式批量迭代
        3. 按 (deliver_tick) 分组，每组一个 payload 向量，消除 N×M 次 np.zeros 分配
        """
        n_spikes = len(spike_indices)
        if n_spikes <= 0:
            return 0
        if self.weight_matrix is None or self.delay_matrix is None:
            return 0

        # 一次性取出所有放电行的子矩阵 (n_spikes, n)
        spike_rows_w = self.weight_matrix[spike_indices]
        spike_rows_d = self.delay_matrix[spike_indices]

        # 转为 COO 批量遍历，coo_w.row 是局部行索引 (0..n_spikes-1)
        coo_w = spike_rows_w.tocoo()
        coo_d = spike_rows_d.tocoo()

        if len(coo_w.data) == 0:
            return 0

        propagation_count = len(coo_w.data)

        # 按 (deliver_tick) 分组累积电流，key: deliver_tick, value: payload 向量
        tick_payloads: dict[int, np.ndarray] = {}

        for edge_idx in range(len(coo_w.data)):
            local_src = coo_w.row[edge_idx]
            target = coo_w.col[edge_idx]
            weight = float(coo_w.data[edge_idx])
            delay = max(0, int(coo_d.data[edge_idx]))

            if not self.alive[target]:
                continue

            # Token 传播：从源神经元复制最新 token 到目标
            src_idx = int(spike_indices[local_src])
            src_mem_len = int(self.token_mem_len[src_idx])
            if src_mem_len > 0:
                mem_size = self.token_memory.shape[1]
                src_ptr = int(self.token_mem_ptr[src_idx])
                src_last = (src_ptr - 1) % mem_size
                propagated_token = int(self.token_memory[src_idx, src_last])
                self._write_token(target, propagated_token)

            deliver_tick = int(tick) + delay
            if deliver_tick not in tick_payloads:
                tick_payloads[deliver_tick] = np.zeros(self.n, dtype=np.float64)
            tick_payloads[deliver_tick][target] += weight

        # 批量调度，每 (deliver_tick) 仅一次 _schedule_currents 调用
        for deliver_tick, payload in tick_payloads.items():
            self._schedule_currents(deliver_tick, payload)

        self.total_propagations += propagation_count
        return propagation_count

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
        if self.weight_matrix is None:
            return np.zeros(self.n, dtype=np.float64)
        return np.asarray(self.weight_matrix[spike_indices].sum(axis=0)).ravel()
