"""性能基准回归测试 — P6.5

覆盖 Network.tick() / VectorizedNetwork.tick() 性能基线、connect_by_proximity()
性能、config.py to_dict() 正确性，以及向量化批处理的结构性操作计数守护
（批量调度次数、动力学内核调用次数，原两个文件合并）。
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from neurogenesis_lab.config import SimConfig
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.vectorized import VectorizedNetwork

# ── 辅助函数 ──────────────────────────────────────────────


def _run_mixed_model_benchmark(neurons: int = 1000, ticks: int = 100) -> dict[str, float | int | str]:
    """运行混合模型基准并返回统一基线格式。"""
    vnet = VectorizedNetwork()
    vnet.init_neurons(neurons, energy_init=1.0)
    # 0=LIF, 1=Izh, 2=AdEx，按 1/3 切分
    third = neurons // 3
    vnet.model_type[:third] = 0
    vnet.model_type[third : 2 * third] = 1
    vnet.model_type[2 * third :] = 2
    vnet.connect_knn(k=3, weight=1.0, delay=1)

    start = time.time()
    for t in range(1, ticks + 1):
        vnet.tick(t, dt=1.0)
    elapsed = time.time() - start
    summary = vnet.get_summary()

    return {
        "scenario": "mixed_model_vectorized_1000x100",
        "neurons": neurons,
        "ticks": ticks,
        "elapsed_s": elapsed,
        "alive_neurons": int(summary["alive_neurons"]),
        "total_spikes": int(summary["total_spikes"]),
    }


class TestNetworkTickPerformance:
    """Network.tick() 性能基准"""

    def test_50_neurons_100_ticks_under_5s(self, ring_network_factory):
        """50 个神经元运行 100 ticks 应在 5 秒内完成"""
        net = ring_network_factory(n=50)

        start = time.time()
        for t in range(1, 101):
            net.tick(t, dt=1.0)
        elapsed = time.time() - start

        assert elapsed < 5.0, (
            f"50 神经元 100 ticks 耗时 {elapsed:.2f}s，超过 5s 阈值"
        )

    def test_tick_returns_valid_snapshot(self, ring_network_factory):
        """tick() 返回的 snapshot 包含必要字段"""
        net = ring_network_factory(n=10)
        snapshot = net.tick(1, dt=1.0)

        required_keys = [
            "tick", "alive_neurons", "total_neurons",
            "tick_spikes", "prediction_error", "mean_energy",
        ]
        for key in required_keys:
            assert key in snapshot, f"snapshot 缺少字段: {key}"


# ══════════════════════════════════════════════════════════
# 2. VectorizedNetwork.tick() 性能基线
# ══════════════════════════════════════════════════════════


class TestVectorizedNetworkTickPerformance:
    """VectorizedNetwork.tick() 性能基准"""

    def test_1000_neurons_100_ticks_under_10s(self):
        """1000 个神经元运行 100 ticks 应在 10 秒内完成"""
        vnet = VectorizedNetwork()
        vnet.init_neurons(1000, energy_init=1.0)
        vnet.connect_knn(k=3, weight=1.0, delay=1)

        start = time.time()
        for t in range(1, 101):
            vnet.tick(t, dt=1.0)
        elapsed = time.time() - start

        assert elapsed < 10.0, (
            f"1000 神经元 100 ticks 耗时 {elapsed:.2f}s，超过 10s 阈值"
        )

    def test_vectorized_tick_returns_valid_snapshot(self):
        """向量化 tick() 返回的 snapshot 包含必要字段"""
        vnet = VectorizedNetwork()
        vnet.init_neurons(100, energy_init=1.0)
        vnet.connect_knn(k=3)

        snapshot = vnet.tick(1, dt=1.0)

        required_keys = [
            "tick", "alive_neurons", "total_neurons",
            "tick_spikes", "prediction_error", "mean_energy",
        ]
        for key in required_keys:
            assert key in snapshot, f"snapshot 缺少字段: {key}"


# ══════════════════════════════════════════════════════════
# 3. connect_by_proximity() 性能
# ══════════════════════════════════════════════════════════


class TestConnectByProximityPerformance:
    """connect_by_proximity() 大规模连接性能"""

    def test_5000_neurons_under_30s(self):
        """5000 个神经元的近邻连接应在 30 秒内完成（宽松阈值，CI 友好）"""
        neurons = [LIFNeuron() for _ in range(5000)]
        net = Network(
            neurons={n.id: n for n in neurons},
            max_neurons=6000,
            energy_cap=5.0,
        )

        start = time.time()
        net.connect_by_proximity(radius=3.0, max_links_per_neuron=5)
        elapsed = time.time() - start

        assert elapsed < 30.0, (
            f"5000 神经元 connect_by_proximity 耗时 {elapsed:.2f}s，超过 30s 阈值"
        )

    def test_connections_actually_created(self):
        """connect_by_proximity 应实际创建连接"""
        neurons = [LIFNeuron() for _ in range(20)]
        # 手动设置位置使部分神经元靠近
        for i, n in enumerate(neurons):
            n.pos = (float(i % 5), float(i // 5))

        net = Network(
            neurons={n.id: n for n in neurons},
            max_neurons=50,
            energy_cap=5.0,
        )
        net.connect_by_proximity(radius=2.0, max_links_per_neuron=3)

        total_links = sum(len(n.links) for n in net.neurons.values())
        assert total_links > 0, "connect_by_proximity 应创建至少一条连接"


# ══════════════════════════════════════════════════════════
# 4. config.py to_dict() 正确性
# ══════════════════════════════════════════════════════════


class TestMixedModelBenchmark:
    """混合模型基准测试（P2.3.4）。"""

    def test_mixed_1000_neurons_100_ticks_under_15s(self):
        """1000 混合神经元 100 ticks 应在 15 秒内完成。"""
        result = _run_mixed_model_benchmark(neurons=1000, ticks=100)
        assert float(result["elapsed_s"]) < 15.0, (
            f"混合模型 1000x100 耗时 {float(result['elapsed_s']):.2f}s，超过 15s 阈值"
        )

    def test_mixed_benchmark_baseline_output_format(self):
        """基线输出应包含稳定字段，便于后续对比与落库。"""
        result = _run_mixed_model_benchmark(neurons=1000, ticks=100)
        expected_keys = {
            "scenario",
            "neurons",
            "ticks",
            "elapsed_s",
            "alive_neurons",
            "total_spikes",
        }
        assert expected_keys <= set(result.keys())
        assert result["scenario"] == "mixed_model_vectorized_1000x100"
        assert int(result["neurons"]) == 1000
        assert int(result["ticks"]) == 100
        assert float(result["elapsed_s"]) >= 0.0


class TestSimConfigToDict:
    """SimConfig.to_dict() 使用 asdict 后的正确性"""

    def test_to_dict_contains_all_top_level_keys(self):
        """to_dict() 返回的字典应包含所有预期的顶层键"""
        cfg = SimConfig()
        d = cfg.to_dict()

        expected_keys = {
            "ticks",
            "seed",
            "dt",
            "neuron",
            "network",
            "evolution",
            "schema_version",
        }
        assert expected_keys.issubset(d.keys()), (
            f"缺少顶层键: {expected_keys - d.keys()}"
        )

    def test_nested_neuron_structure(self):
        """嵌套的 neuron 子字典应包含正确的键"""
        cfg = SimConfig()
        d = cfg.to_dict()

        neuron_keys = {
            "v_rest",
            "v_reset",
            "v_th",
            "tau_m",
            "r_m",
            "spike_cost",
            "input_gain",
            "mem_size",
        }
        assert neuron_keys.issubset(d["neuron"].keys()), (
            f"neuron 子字典缺少键: {neuron_keys - set(d['neuron'].keys())}"
        )

    def test_nested_network_structure(self):
        """嵌套的 network 子字典应包含正确的键"""
        cfg = SimConfig()
        d = cfg.to_dict()

        network_keys = {
            "reproduction_threshold",
            "reproduction_share",
            "mutation_scale",
            "energy_decay",
            "max_neurons",
            "max_links",
            "corpse_ttl",
            "scavenger_radius",
            "energy_cap",
        }
        assert network_keys.issubset(d["network"].keys()), (
            f"network 子字典缺少键: {network_keys - set(d['network'].keys())}"
        )

    def test_nested_evolution_structure(self):
        """嵌套的 evolution 子字典应包含正确的键"""
        cfg = SimConfig()
        d = cfg.to_dict()

        evolution_keys = {"fitness_weights", "mutation_amplitude", "min_neurons"}
        assert evolution_keys.issubset(d["evolution"].keys()), (
            f"evolution 子字典缺少键: {evolution_keys - set(d['evolution'].keys())}"
        )

    def test_to_dict_values_match_defaults(self):
        """to_dict() 的值应与默认配置一致"""
        cfg = SimConfig()
        d = cfg.to_dict()

        assert d["ticks"] == 100
        assert d["seed"] == 42
        assert d["dt"] == 1.0
        assert d["neuron"]["v_rest"] == -65.0
        assert d["neuron"]["v_th"] == -55.0
        assert d["network"]["max_neurons"] == 50
        assert d["evolution"]["min_neurons"] == 2

    def test_to_dict_with_custom_values(self):
        """自定义配置后 to_dict() 应反映修改"""
        from neurogenesis_lab.config import EvolutionParams, NetworkParams, NeuronParams

        cfg = SimConfig(
            ticks=200,
            seed=99,
            neuron=NeuronParams(v_th=-50.0),
            network=NetworkParams(max_neurons=100),
            evolution=EvolutionParams(min_neurons=5),
        )
        d = cfg.to_dict()

        assert d["ticks"] == 200
        assert d["seed"] == 99
        assert d["neuron"]["v_th"] == -50.0
        assert d["network"]["max_neurons"] == 100
        assert d["evolution"]["min_neurons"] == 5

    def test_to_dict_is_plain_dict(self):
        """to_dict() 返回的应是纯 dict，不含 dataclass 实例"""
        cfg = SimConfig()
        d = cfg.to_dict()

        assert isinstance(d, dict)
        assert isinstance(d["neuron"], dict)
        assert isinstance(d["network"], dict)
        assert isinstance(d["evolution"], dict)


# ======================================================================
# 合并自 test_performance_regression.py：性能回归测试 — 守护 VectorizedNetwork 向量化优化不退化。
# ======================================================================


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
