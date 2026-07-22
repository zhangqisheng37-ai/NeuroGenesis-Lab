"""性能回归测试 — 守护 VectorizedNetwork 向量化优化不退化。

不同于 test_performance.py 的挂钟阈值（易受机器/CI 抖动影响），本模块使用
**确定性操作计数断言**，直接锁定向量化优化的结构性不变量：

1. ``_propagate_spikes`` 按投递 tick 分组批量调度：``_schedule_currents`` 调用次数
   等于「不同投递 tick 数」（O(不同延迟)），而非边数（O(边数)）。
2. 批量 numpy 动力学：每 tick 每种模型的 step 内核最多调用一次（防退化为逐神经元循环）。
3. 传播正确性：批量分组调度产生的电流与朴素逐边累加参考值一致。

这些断言在后续重构中若被破坏（例如恢复为逐边 ``np.zeros`` 分配或逐神经元 Python 循环），
将立即失败。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from neurogenesis_lab.vectorized import VectorizedNetwork


def _install_call_counter(monkeypatch, obj: object, name: str) -> dict[str, int]:
    """用计数包装实例方法，返回记录调用次数的字典。"""
    counter = {"count": 0}
    original = getattr(obj, name)

    def _wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        counter["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, name, _wrapper)
    return counter


def _set_edges(
    vnet: VectorizedNetwork,
    rows: list[int],
    cols: list[int],
    weights: list[float],
    delays: list[int],
) -> None:
    """直接构造权重/延迟稀疏矩阵，绕过 connect_* 以精确控制边与延迟。"""
    shape = (vnet.n, vnet.n)
    vnet.weight_matrix = csr_matrix(
        (np.asarray(weights, dtype=np.float64), (rows, cols)), shape=shape
    )
    vnet.delay_matrix = csr_matrix(
        (np.asarray(delays, dtype=np.int32), (rows, cols)), shape=shape, dtype=np.int32
    )


# ══════════════════════════════════════════════════════════
# 1. _propagate_spikes 按延迟分组批量调度
# ══════════════════════════════════════════════════════════


class TestPropagateSpikesBatching:
    """守护 _propagate_spikes 的「按投递 tick 分组」优化。"""

    def test_propagate_spikes_batches_schedule_calls(self, monkeypatch) -> None:
        """均匀延迟下，无论边数多少，_schedule_currents 只应调用一次。"""
        vnet = VectorizedNetwork()
        vnet.init_neurons(50, energy_init=1.0)

        # 3 个放电源 × 47 个目标 = 141 条边，全部 delay=1 → 仅 1 个投递 tick
        rows: list[int] = []
        cols: list[int] = []
        weights: list[float] = []
        delays: list[int] = []
        for src in (0, 1, 2):
            for tgt in range(3, 50):
                rows.append(src)
                cols.append(tgt)
                weights.append(0.5)
                delays.append(1)
        _set_edges(vnet, rows, cols, weights, delays)

        counter = _install_call_counter(monkeypatch, vnet, "_schedule_currents")
        propagated = vnet._propagate_spikes(tick=1, spike_indices=np.array([0, 1, 2]))

        assert propagated == len(rows), "应统计全部传播边数"
        assert counter["count"] == 1, (
            f"均匀延迟下 141 条边应仅批量调度 1 次，实际 {counter['count']} 次 "
            "— 疑似退化为逐边/逐目标调度"
        )

    def test_propagate_spikes_scales_with_distinct_delays(self, monkeypatch) -> None:
        """K 种不同延迟 → _schedule_currents 恰调用 K 次（O(不同延迟)）。"""
        vnet = VectorizedNetwork()
        vnet.init_neurons(10, energy_init=1.0)

        k = 5
        rows = [0] * k
        cols = list(range(1, k + 1))
        weights = [1.0] * k
        delays = list(range(1, k + 1))  # 1..5，互不相同
        _set_edges(vnet, rows, cols, weights, delays)

        counter = _install_call_counter(monkeypatch, vnet, "_schedule_currents")
        vnet._propagate_spikes(tick=1, spike_indices=np.array([0]))

        assert counter["count"] == k, (
            f"{k} 种不同延迟应批量调度 {k} 次，实际 {counter['count']} 次"
        )

    def test_propagation_currents_equivalence(self) -> None:
        """批量分组调度的电流应与朴素逐边累加参考值完全一致。"""
        vnet = VectorizedNetwork()
        vnet.init_neurons(20, energy_init=1.0)

        rows = [0, 0, 1, 1, 2]
        cols = [3, 4, 4, 5, 6]
        weights = [0.5, 1.0, 0.25, 0.75, 2.0]
        delays = [1, 2, 1, 2, 1]
        _set_edges(vnet, rows, cols, weights, delays)

        tick = 1
        vnet._propagate_spikes(tick=tick, spike_indices=np.array([0, 1, 2]))

        # 朴素参考：逐边把权重累加到 (tick+delay) 的目标电流
        reference: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(vnet.n, dtype=np.float64)
        )
        for r, c, w, d in zip(rows, cols, weights, delays):
            _ = r
            reference[tick + d][c] += w

        assert set(vnet.scheduled_inputs.keys()) == set(reference.keys())
        for deliver_tick, payload in reference.items():
            assert np.allclose(vnet.scheduled_inputs[deliver_tick], payload), (
                f"投递 tick {deliver_tick} 的电流与参考值不一致"
            )


# ══════════════════════════════════════════════════════════
# 2. 批量 numpy 动力学：每种模型的 step 内核每 tick 仅调用一次
# ══════════════════════════════════════════════════════════


class TestVectorizedDynamicsBatching:
    """守护多模型动力学的向量化批处理（防退化为逐神经元循环）。"""

    def test_lif_step_called_once_per_tick(self, monkeypatch) -> None:
        """全 LIF 网络：一次 tick 内 _lif_step 应恰调用 1 次。"""
        vnet = VectorizedNetwork()
        vnet.init_neurons(200, energy_init=1.0)  # 默认全 LIF (model_type=0)
        vnet.connect_knn(k=3, weight=1.0, delay=1)

        counter = _install_call_counter(monkeypatch, vnet, "_lif_step")
        vnet.tick(1, dt=1.0)

        assert counter["count"] == 1, (
            f"200 个 LIF 神经元一次 tick 应批量计算 1 次，实际 {counter['count']} 次 "
            "— 疑似退化为逐神经元循环"
        )

    def test_mixed_model_dispatch_bounded(self, monkeypatch) -> None:
        """混合模型：三种 step 内核每 tick 各恰调用 1 次（按 model_type 掩码分组）。"""
        vnet = VectorizedNetwork()
        vnet.init_neurons(99, energy_init=1.0)
        vnet.model_type[:33] = 0  # LIF
        vnet.model_type[33:66] = 1  # Izhikevich
        vnet.model_type[66:] = 2  # AdEx
        vnet.connect_knn(k=3, weight=1.0, delay=1)

        lif = _install_call_counter(monkeypatch, vnet, "_lif_step")
        izh = _install_call_counter(monkeypatch, vnet, "_izh_step")
        adex = _install_call_counter(monkeypatch, vnet, "_adex_step")
        vnet.tick(1, dt=1.0)

        assert lif["count"] == 1, f"_lif_step 应调用 1 次，实际 {lif['count']}"
        assert izh["count"] == 1, f"_izh_step 应调用 1 次，实际 {izh['count']}"
        assert adex["count"] == 1, f"_adex_step 应调用 1 次，实际 {adex['count']}"
