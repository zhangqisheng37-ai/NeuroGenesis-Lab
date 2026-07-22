"""向量化多模型动力学 — vectorized 子包的 LIF/Izhikevich/AdEx 步进部分。"""

from __future__ import annotations

import numpy as np

from neurogenesis_lab.vectorized.state import VectorizedState


class DynamicsMixin(VectorizedState):
    """输入注入与多模型批量动力学步进。"""

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
            combined_inputs = np.nan_to_num(combined_inputs, nan=0.0, posinf=0.0, neginf=0.0)
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
        exp_arg = np.clip((v - self.v_th[mask]) / np.maximum(delta_t, 1e-6), -50.0, 50.0)
        dv = (
            -(v - self.v_rest[mask])
            + delta_t * np.exp(exp_arg)
            - self.r_m[mask] * u
            + self.r_m[mask] * I
        ) / self.tau_m[mask]
        du = (self.adex_a[mask] * (v - self.v_rest[mask]) - u) / self.adex_tau_w[mask]
        self.v[mask] += dv * dt
        self.u[mask] += du * dt
