"""NeuroGenesis Lab — 统一 CLI 入口

用法:
    python -m neurogenesis_lab run [--ticks N] [--seed S] [--output DIR]
    python -m neurogenesis_lab run --config experiment.toml
    python -m neurogenesis_lab benchmark [--neurons N] [--ticks T]
    python -m neurogenesis_lab info
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from neurogenesis_lab import __version__
from neurogenesis_lab.config import (
    SCHEMA_VERSION,
    EvolutionParams,
    NetworkParams,
    NeuronParams,
    SimConfig,
    seed_all,
)
from neurogenesis_lab.core_bridge import parallel_lif_tick_safe, rust_core_status
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.learning import AutoLearner, RuleDiscovery
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron, IzhikevichNeuron, AdExNeuron
from neurogenesis_lab.observability import (
    NetworkObserver,
    ObservabilityLogger,
    create_experiment_logger,
)


def _load_toml_config(path: Path) -> dict:
    """加载 TOML 配置文件"""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore
    with open(path, "rb") as f:
        return tomllib.load(f)


def _build_sim_config(args: argparse.Namespace) -> SimConfig:
    """从命令行参数和可选 TOML 文件构建配置"""
    cfg = SimConfig()

    def _section(data: dict, name: str) -> dict:
        value = data.get(name, {})
        return value if isinstance(value, dict) else {}

    def _float_map(values: dict[str, object], allowed: set[str]) -> dict[str, float]:
        mapped: dict[str, float] = {}
        for key, val in values.items():
            if key in allowed and isinstance(val, (int, float)):
                mapped[key] = float(val)
        return mapped

    if hasattr(args, "config") and args.config:
        toml_data = _load_toml_config(Path(args.config))
        sim = _section(toml_data, "simulation")
        neuron = _section(toml_data, "neuron")
        network = _section(toml_data, "network")
        evolution = _section(toml_data, "evolution")
        fitness_weights = _section(evolution, "fitness_weights")

        cfg = SimConfig(
            ticks=int(sim.get("ticks", cfg.ticks)),
            seed=int(sim.get("seed", cfg.seed)),
            dt=float(sim.get("dt", cfg.dt)),
            neuron=NeuronParams(
                v_rest=float(neuron.get("v_rest", cfg.neuron.v_rest)),
                v_reset=float(neuron.get("v_reset", cfg.neuron.v_reset)),
                v_th=float(neuron.get("v_th", cfg.neuron.v_th)),
                tau_m=float(neuron.get("tau_m", cfg.neuron.tau_m)),
                r_m=float(neuron.get("r_m", cfg.neuron.r_m)),
                spike_cost=float(neuron.get("spike_cost", cfg.neuron.spike_cost)),
                input_gain=float(neuron.get("input_gain", cfg.neuron.input_gain)),
                mem_size=int(neuron.get("mem_size", cfg.neuron.mem_size)),
            ),
            network=NetworkParams(
                reproduction_threshold=float(
                    network.get(
                        "reproduction_threshold", cfg.network.reproduction_threshold
                    )
                ),
                reproduction_share=float(
                    network.get("reproduction_share", cfg.network.reproduction_share)
                ),
                mutation_scale=float(
                    network.get("mutation_scale", cfg.network.mutation_scale)
                ),
                energy_decay=float(
                    network.get("energy_decay", cfg.network.energy_decay)
                ),
                max_neurons=int(network.get("max_neurons", cfg.network.max_neurons)),
                max_links=int(network.get("max_links", cfg.network.max_links)),
                corpse_ttl=int(network.get("corpse_ttl", cfg.network.corpse_ttl)),
                scavenger_radius=float(
                    network.get("scavenger_radius", cfg.network.scavenger_radius)
                ),
                input_current_scale=float(
                    network.get("input_current_scale", cfg.network.input_current_scale)
                ),
                background_current=float(
                    network.get("background_current", cfg.network.background_current)
                ),
                spontaneous_spike_rate=float(
                    network.get(
                        "spontaneous_spike_rate", cfg.network.spontaneous_spike_rate
                    )
                ),
                low_activation_threshold=int(
                    network.get(
                        "low_activation_threshold",
                        cfg.network.low_activation_threshold,
                    )
                ),
                move_cost=float(network.get("move_cost", cfg.network.move_cost)),
                energy_cap=float(network.get("energy_cap", cfg.network.energy_cap)),
                adaptive_decay=bool(
                    network.get("adaptive_decay", cfg.network.adaptive_decay)
                ),
                adaptive_reproduction=bool(
                    network.get(
                        "adaptive_reproduction", cfg.network.adaptive_reproduction
                    )
                ),
                endangered_threshold=int(
                    network.get(
                        "endangered_threshold", cfg.network.endangered_threshold
                    )
                ),
            ),
            evolution=EvolutionParams(
                fitness_weights={
                    **cfg.evolution.fitness_weights,
                    **_float_map(
                        fitness_weights,
                        allowed={"propagation", "energy_efficiency", "prediction_hit"},
                    ),
                },
                mutation_amplitude=float(
                    evolution.get(
                        "mutation_amplitude", cfg.evolution.mutation_amplitude
                    )
                ),
                energy_budget_per_tick=float(
                    evolution.get(
                        "energy_budget_per_tick", cfg.evolution.energy_budget_per_tick
                    )
                ),
                congestion_penalty=float(
                    evolution.get(
                        "congestion_penalty", cfg.evolution.congestion_penalty
                    )
                ),
                min_neurons=int(
                    evolution.get("min_neurons", cfg.evolution.min_neurons)
                ),
            ),
        )
    # 命令行覆盖
    if hasattr(args, "ticks") and args.ticks is not None:
        cfg.ticks = args.ticks
    if hasattr(args, "seed") and args.seed is not None:
        cfg.seed = args.seed
    if hasattr(args, "dt") and args.dt is not None:
        cfg.dt = args.dt
    if hasattr(args, "background_current") and args.background_current is not None:
        cfg.network.background_current = float(args.background_current)
    if (
        hasattr(args, "spontaneous_spike_rate")
        and args.spontaneous_spike_rate is not None
    ):
        cfg.network.spontaneous_spike_rate = float(args.spontaneous_spike_rate)
    if (
        hasattr(args, "low_activation_threshold")
        and args.low_activation_threshold is not None
    ):
        cfg.network.low_activation_threshold = int(args.low_activation_threshold)
    return cfg


def cmd_run(args: argparse.Namespace) -> None:
    """运行仿真实验"""
    cfg = _build_sim_config(args)
    errors = cfg.validate()
    if errors:
        print("配置验证失败:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    seed_all(cfg.seed)
    print(f"[NeuroGenesis Lab v{__version__}] 运行仿真")
    print(f"  ticks={cfg.ticks}, seed={cfg.seed}, schema={SCHEMA_VERSION}")

    resume_path = getattr(args, "resume_from", None)
    if resume_path:
        network = Network.load_checkpoint(Path(resume_path))
        run_start_tick = network.current_tick + 1
        run_end_tick = network.current_tick + cfg.ticks
        print(
            f"  已从 checkpoint 恢复: {resume_path} (from tick={network.current_tick})"
        )
    else:
        # 创建网络
        neuron_count = getattr(args, "neurons", 8) or 8
        neuron_types = [LIFNeuron, IzhikevichNeuron, AdExNeuron]
        common_neuron_kwargs = {
            "v_rest": cfg.neuron.v_rest,
            "v_reset": cfg.neuron.v_reset,
            "v_th": cfg.neuron.v_th,
            "tau_m": cfg.neuron.tau_m,
            "r_m": cfg.neuron.r_m,
            "spike_cost": cfg.neuron.spike_cost,
            "input_gain": cfg.neuron.input_gain,
            "mem_size": cfg.neuron.mem_size,
            "move_cost": cfg.network.move_cost,
        }
        neurons = [
            neuron_types[i % len(neuron_types)](**common_neuron_kwargs)
            for i in range(neuron_count)
        ]
        network = Network(
            neurons={n.id: n for n in neurons},
            reproduction_threshold=cfg.network.reproduction_threshold,
            reproduction_share=cfg.network.reproduction_share,
            mutation_scale=cfg.network.mutation_scale,
            energy_decay=cfg.network.energy_decay,
            max_neurons=cfg.network.max_neurons,
            max_links=cfg.network.max_links,
            corpse_ttl=cfg.network.corpse_ttl,
            scavenger_radius=cfg.network.scavenger_radius,
            input_current_scale=cfg.network.input_current_scale,
            background_current=cfg.network.background_current,
            spontaneous_spike_rate=cfg.network.spontaneous_spike_rate,
            move_cost=cfg.network.move_cost,
            energy_cap=cfg.network.energy_cap,
            adaptive_decay=cfg.network.adaptive_decay,
            adaptive_reproduction=cfg.network.adaptive_reproduction,
            endangered_threshold=cfg.network.endangered_threshold,
            fitness_weights=cfg.evolution.fitness_weights,
            congestion_penalty=cfg.evolution.congestion_penalty,
            min_neurons=cfg.evolution.min_neurons,
            evolution_mutation_amplitude=cfg.evolution.mutation_amplitude,
            energy_budget_per_tick=cfg.evolution.energy_budget_per_tick,
        )
        # 环形连接
        for i, neuron in enumerate(neurons):
            target = neurons[(i + 1) % len(neurons)]
            network.connect(neuron.id, target.id, weight=1.0, delay=1)

        # 初始空间连接
        network.connect_by_proximity(radius=5.0, max_links_per_neuron=3)

        # 注入信号
        encoder = Encoder()
        expr = getattr(args, "expr", "1+2=3") or "1+2=3"
        input_weight = max(0.0, float(getattr(args, "input_weight", 40.0) or 40.0))
        stim_interval = max(1, int(getattr(args, "stim_interval", 5) or 5))
        tokens = encoder.encode_expr(expr)
        for tick_start in range(1, cfg.ticks + 1, stim_interval):
            base_events = encoder.to_events(tokens, src_id=neurons[0].id)
            scheduled = [
                Event(
                    token_id=e.token_id,
                    t=e.t + tick_start,
                    src_id=e.src_id,
                    meta={**e.meta, "weight": input_weight},
                )
                for e in base_events
                if e.t + tick_start <= cfg.ticks
            ]
            network.enqueue(scheduled)
        run_start_tick = 1
        run_end_tick = cfg.ticks

    # 输出设置
    output_dir = Path(getattr(args, "output", "outputs") or "outputs")
    logger = create_experiment_logger(
        output_dir,
        "run",
        debug_mode=bool(getattr(args, "debug", False)),
        log_interval=max(1, int(getattr(args, "log_interval", 1) or 1)),
        max_file_size=max(0, int(getattr(args, "max_file_size", 0) or 0)),
        gzip_output=bool(getattr(args, "gzip_output", False)),
        async_mode=bool(getattr(args, "async_log", False)),
        log_buffer_size=max(1, int(getattr(args, "log_buffer_size", 256) or 256)),
    )
    debug_neuron_ids = []
    if network.neurons:
        debug_neuron_ids = [next(iter(network.neurons.keys()))]
    observer = NetworkObserver(network, debug_neuron_ids=debug_neuron_ids)
    logger.log_config(cfg.to_dict())

    # 自动学习引擎
    auto_learner = AutoLearner(discovery=RuleDiscovery(min_support=2))

    # 实时可视化监控
    live_monitor = None
    if getattr(args, "visualize", False):
        try:
            from neurogenesis_lab.visualization import LiveMonitor

            plot_interval = max(1, int(getattr(args, "plot_interval", 5) or 5))
            live_monitor = LiveMonitor(plot_interval=plot_interval)
            print(f"  实时监控已启用 (更新间隔: {plot_interval} ticks)")
        except Exception as exc:
            print(f"  警告: 无法启动实时监控 ({exc})")

    checkpoint_interval = max(0, int(getattr(args, "checkpoint_interval", 0) or 0))
    checkpoint_dir = Path(
        getattr(args, "checkpoint_dir", str(output_dir)) or str(output_dir)
    )

    # 运行
    t0 = time.time()
    total_ticks = run_end_tick - run_start_tick + 1
    for t in range(run_start_tick, run_end_tick + 1):
        snapshot = network.tick(t, dt=cfg.dt)
        # 自动学习 — 采样、规则发现、预测奖励
        learn_report = auto_learner.on_tick(t, snapshot, network)
        snapshot["rule_count"] = learn_report["rule_count"]
        snapshot["prediction_accuracy"] = learn_report["prediction_accuracy"]
        logger.log_tick(observer, t, snapshot)
        local_progress_tick = t - run_start_tick + 1
        if checkpoint_interval > 0 and local_progress_tick % checkpoint_interval == 0:
            cp_path = checkpoint_dir / f"checkpoint_tick_{t}.pkl"
            network.save_checkpoint(cp_path, tick=t)
        if live_monitor is not None:
            live_monitor.on_tick(t, snapshot, network)
        if local_progress_tick % max(1, total_ticks // 5) == 0:
            alive = snapshot["alive_neurons"]
            pe = snapshot["prediction_error"]
            acc = learn_report["prediction_accuracy"]
            rules = learn_report["rule_count"]
            print(
                f"  tick {local_progress_tick}/{total_ticks} (global={t}): alive={alive}, PE={pe:.4f}, rules={rules}, acc={acc:.3f}"
            )

    elapsed = time.time() - t0
    if live_monitor is not None:
        live_monitor.finalize()
    final = observer.get_network_metrics(run_end_tick)
    summary = network.get_summary()
    logger.log_summary(run_end_tick, final)
    logger.close()

    print(f"\n完成! 用时 {elapsed:.2f}s")
    print(f"  输出: {logger.get_output_path()}")
    print(f"  存活: {summary['alive_neurons']}/{summary['total_neurons']}")
    print(
        f"  总放电: {summary['total_spikes']}, 总传播: {summary['total_propagations']}"
    )
    print(f"  出生: {summary['birth_count']}, 死亡: {summary['death_count']}")
    learn_summary = auto_learner.get_summary()
    print(
        f"  规则数: {learn_summary['rule_count']}, 预测准确率: {learn_summary['prediction_accuracy']:.3f}"
    )
    print(
        f"  预测次数: {learn_summary['total_predictions']}, 正确: {learn_summary['correct_predictions']}"
    )
    low_activation_threshold = max(0, int(cfg.network.low_activation_threshold))
    if (
        summary["total_spikes"] <= low_activation_threshold
        or summary["total_propagations"] <= low_activation_threshold
    ):
        print(
            f"  警告: 低激活（阈值={low_activation_threshold}），可提高 --input-weight 或调整 [neuron]/[network] 参数"
        )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """性能基准测试"""
    import numpy as np
    from neurogenesis_lab.vectorized import VectorizedNetwork

    neuron_count = getattr(args, "neurons", 1000) or 1000
    tick_count = getattr(args, "ticks", 100) or 100
    seed = getattr(args, "seed", 42) or 42
    rust_size = max(8, int(getattr(args, "rust_size", 10000) or 10000))

    print(f"[NeuroGenesis Lab] 性能基准测试")
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

    # --- Rust 并行 LIF 微基准（含 Python 降级）---
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
            "vectorized": "LIF",
            "comparable": "不可比",
            "note": "动力学细节模型不同",
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
        f"  向量化引擎 ({neuron_count} neurons): {vec_time:.3f}s ({tick_count / vec_time:.0f} ticks/s)"
    )
    print(
        f"  原始引擎 ({compare_n} neurons): {orig_time:.3f}s ({tick_count / orig_time:.0f} ticks/s)"
    )
    print(
        f"  Rust并行LIF ({rust_size} neurons, backend={rust_backend}): {rust_time:.3f}s ({tick_count / rust_time:.0f} ticks/s), spikes={rust_spikes}"
    )
    if compare_n == neuron_count and orig_time > 0:
        print(f"  加速比: {orig_time / vec_time:.1f}x")

    print("\n对照指标表:")
    print(f"  {_line(headers)}")
    print(f"  {_line(['-' * w for w in widths])}")
    for row in table_data:
        print(f"  {_line([str(x) for x in row])}")


def cmd_info(args: argparse.Namespace) -> None:
    """显示版本与配置信息"""
    print(f"NeuroGenesis Lab v{__version__}")
    print(f"Schema Version: {SCHEMA_VERSION}")
    cfg = SimConfig()
    rust_status = rust_core_status()
    enabled_components = [
        name for name, enabled in rust_status.get("components", {}).items() if enabled
    ]
    print(f"Rust Core: {'available' if rust_status['available'] else 'unavailable'}")
    if enabled_components:
        print("Rust Components:")
        for comp in enabled_components:
            print(f"  - {comp}")
    print(f"默认配置:")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neurogenesis-lab",
        description="NeuroGenesis Lab — 数字生命仿真平台",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run
    run_parser = subparsers.add_parser("run", help="运行仿真实验")
    run_parser.add_argument("--ticks", type=int, default=None, help="仿真步数")
    run_parser.add_argument("--seed", type=int, default=None, help="随机种子")
    run_parser.add_argument("--dt", type=float, default=None, help="时间步长")
    run_parser.add_argument("--neurons", type=int, default=8, help="初始神经元数")
    run_parser.add_argument("--expr", type=str, default="1+2=3", help="输入表达式")
    run_parser.add_argument(
        "--input-weight",
        type=float,
        default=40.0,
        help="外部注入事件的默认权重（用于激活动力学）",
    )
    run_parser.add_argument(
        "--stim-interval",
        type=int,
        default=5,
        help="外部刺激注入间隔（tick）",
    )
    run_parser.add_argument(
        "--background-current",
        type=float,
        default=None,
        help="全局背景电流（每个神经元每 tick 注入）",
    )
    run_parser.add_argument(
        "--spontaneous-spike-rate",
        type=float,
        default=None,
        help="自发放电概率（每神经元每 tick，范围 [0,1]）",
    )
    run_parser.add_argument(
        "--low-activation-threshold",
        type=int,
        default=None,
        help="低激活告警阈值（总放电/总传播任一不高于该值则告警）",
    )
    run_parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    run_parser.add_argument(
        "--config", type=str, default=None, help="TOML 配置文件路径"
    )
    run_parser.add_argument("--debug", action="store_true", help="开启 debug 模式")
    run_parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="日志采样间隔（每 N tick 记录一次）",
    )
    run_parser.add_argument(
        "--gzip-output",
        action="store_true",
        help="将输出文件写为 .jsonl.gz",
    )
    run_parser.add_argument(
        "--max-file-size",
        type=int,
        default=0,
        help="单个输出文件最大字节数，>0 时启用轮转",
    )
    run_parser.add_argument(
        "--async-log",
        action="store_true",
        help="启用异步双缓冲日志写入，降低 I/O 对主循环阻塞",
    )
    run_parser.add_argument(
        "--log-buffer-size",
        type=int,
        default=256,
        help="异步日志双缓冲阈值（记录条数）",
    )
    run_parser.add_argument(
        "--visualize",
        action="store_true",
        help="启动实时可视化监控面板",
    )
    run_parser.add_argument(
        "--plot-interval",
        type=int,
        default=5,
        help="实时图表更新间隔（每 N tick 更新一次）",
    )
    run_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="checkpoint 保存间隔（每 N 个本次运行 tick 保存一次，0=禁用）",
    )
    run_parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="从 checkpoint 文件恢复运行",
    )
    run_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs",
        help="checkpoint 输出目录",
    )

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="性能基准测试")
    bench_parser.add_argument("--neurons", type=int, default=1000, help="神经元数量")
    bench_parser.add_argument("--ticks", type=int, default=100, help="仿真步数")
    bench_parser.add_argument("--seed", type=int, default=42, help="随机种子")
    bench_parser.add_argument(
        "--rust-size",
        type=int,
        default=10000,
        help="Rust 并行 LIF 微基准神经元数量",
    )

    # info
    subparsers.add_parser("info", help="显示版本与配置信息")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
