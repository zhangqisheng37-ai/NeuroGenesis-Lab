"""CLI benchmark 命令 — 性能基准测试"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from neurogenesis_lab.core_bridge import parallel_lif_tick_safe
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.vectorized import VectorizedNetwork


def cmd_benchmark(args: argparse.Namespace) -> None:
    """性能基准测试。"""
    neuron_count = getattr(args, "neurons", 1000) or 1000
    tick_count = getattr(args, "ticks", 100) or 100
    seed = getattr(args, "seed", 42) or 42
    rust_size = max(8, int(getattr(args, "rust_size", 10000) or 10000))

    print("[NeuroGenesis Lab] 性能基准测试")
    print(f"  neurons={neuron_count}, ticks={tick_count}, seed={seed}")
    print(f"  rust_lif_size={rust_size}")

    # --- 向量化引擎 ---
    np.random.seed(seed)
    vnet = VectorizedNetwork(max_neurons=neuron_count * 2, min_neurons=2)
    vnet.init_neurons(neuron_count, energy_init=1.5)
    vnet.connect_knn(k=3)

    t0 = time.time()
    for t in range(1, tick_count + 1):
        vnet.tick(t)
    vec_time = time.time() - t0
    vec_summary = vnet.get_summary()

    # --- 原始引擎 (较小规模对比) ---
    compare_n = min(neuron_count, 50)
    random.seed(seed)
    neurons = [LIFNeuron() for _ in range(compare_n)]
    net = Network(neurons={n.id: n for n in neurons}, max_neurons=compare_n * 2)
    for i, neuron in enumerate(neurons):
        for j in range(1, min(4, compare_n)):
            target = neurons[(i + j) % compare_n]
            net.connect(neuron.id, target.id, weight=1.0, delay=1)

    t0 = time.time()
    for t in range(1, tick_count + 1):
        net.tick(t)
    orig_time = time.time() - t0
    orig_summary = net.get_summary()

    # --- Rust 并行 LIF 微基准 ---
    rng = np.random.default_rng(seed)
    voltages = rng.normal(loc=-65.0, scale=2.0, size=rust_size).astype(float).tolist()
    inputs = rng.normal(loc=0.5, scale=1.0, size=rust_size).astype(float).tolist()
    rust_dt = 1.0
    rust_t0 = time.time()
    rust_backend = "unknown"
    rust_spikes = 0
    rust_v = voltages
    for _ in range(tick_count):
        rust_v, spikes, rust_backend = parallel_lif_tick_safe(
            rust_v,
            v_rest=-65.0,
            v_reset=-65.0,
            v_th=-55.0,
            tau_m=20.0,
            r_m=1.0,
            inputs=inputs,
            dt=rust_dt,
        )
        rust_spikes += sum(1 for s in spikes if s)
    rust_time = time.time() - rust_t0

    _print_benchmark_results(
        neuron_count=neuron_count,
        compare_n=compare_n,
        tick_count=tick_count,
        vec_time=vec_time,
        orig_time=orig_time,
        rust_time=rust_time,
        rust_size=rust_size,
        rust_backend=rust_backend,
        rust_spikes=rust_spikes,
        vec_summary=vec_summary,
        orig_summary=orig_summary,
    )


def _print_benchmark_results(
    *,
    neuron_count: int,
    compare_n: int,
    tick_count: int,
    vec_time: float,
    orig_time: float,
    rust_time: float,
    rust_size: int,
    rust_backend: str,
    rust_spikes: int,
    vec_summary: dict[str, object],
    orig_summary: dict[str, object],
) -> None:
    """格式化并输出基准测试结果。"""

    def _fmt_num(value: float | int | None, digits: int = 3) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    comparison_rows = [
        {
            "metric": "alive_neurons",
            "original": _fmt_num(orig_summary.get("alive_neurons")),
            "vectorized": _fmt_num(vec_summary.get("alive_neurons")),
            "comparable": "可比",
            "note": "存活数量可直接比较",
        },
        {
            "metric": "total_spikes",
            "original": _fmt_num(orig_summary.get("total_spikes")),
            "vectorized": _fmt_num(vec_summary.get("total_spikes")),
            "comparable": "可比",
            "note": "总放电次数可比较总体趋势",
        },
        {
            "metric": "mean_energy",
            "original": _fmt_num(orig_summary.get("mean_energy"), digits=4),
            "vectorized": _fmt_num(vec_summary.get("mean_energy"), digits=4),
            "comparable": "可比",
            "note": "平均能量可比较总体能量态",
        },
        {
            "metric": "neuron_dynamics_model",
            "original": "LIF + Izh + AdEx",
            "vectorized": "LIF + Izh + AdEx",
            "comparable": "可比",
            "note": "双引擎均支持多模型动力学",
        },
        {
            "metric": "event_routing_trace",
            "original": "逐事件对象路由",
            "vectorized": "向量化延迟路由",
            "comparable": "部分可比",
            "note": "可比较延迟行为，不比较逐事件轨迹",
        },
    ]

    headers = ["指标", "原始引擎", "向量化引擎", "可比性", "说明"]
    table_data = [
        [
            row["metric"],
            row["original"],
            row["vectorized"],
            row["comparable"],
            row["note"],
        ]
        for row in comparison_rows
    ]
    widths = [
        max(len(str(headers[idx])), *(len(str(item[idx])) for item in table_data))
        for idx in range(len(headers))
    ]

    def _line(parts: list[str]) -> str:
        return " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))

    print(f"\n结果:")
    print(
        f"  向量化引擎 ({neuron_count} neurons): "
        f"{vec_time:.3f}s ({tick_count / vec_time:.0f} ticks/s)"
    )
    print(
        f"  原始引擎 ({compare_n} neurons): "
        f"{orig_time:.3f}s ({tick_count / orig_time:.0f} ticks/s)"
    )
    print(
        f"  Rust并行LIF ({rust_size} neurons, backend={rust_backend}): "
        f"{rust_time:.3f}s ({tick_count / rust_time:.0f} ticks/s), spikes={rust_spikes}"
    )
    if compare_n == neuron_count and orig_time > 0:
        print(f"  加速比: {orig_time / vec_time:.1f}x")

    print("\n对照指标表:")
    print(f"  {_line(headers)}")
    print(f"  {_line(['-' * w for w in widths])}")
    for row in table_data:
        print(f"  {_line([str(x) for x in row])}")
