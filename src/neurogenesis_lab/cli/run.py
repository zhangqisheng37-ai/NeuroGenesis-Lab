"""CLI run 命令 — 仿真实验运行"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neurogenesis_lab.visualization import LiveMonitor

from neurogenesis_lab import __version__
from neurogenesis_lab.cli.config_builder import build_sim_config, validate_run_config
from neurogenesis_lab.config import SCHEMA_VERSION, SimConfig, seed_all
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.learning import AutoLearner, RuleDiscovery
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import AdExNeuron, IzhikevichNeuron, LIFNeuron
from neurogenesis_lab.observability import (
    NetworkObserver,
    ObservabilityLogger,
    create_experiment_logger,
)
from neurogenesis_lab.protocols import EngineSummary


def _init_network(cfg: SimConfig, args: argparse.Namespace) -> tuple[Network, int, int]:
    """初始化网络实例和运行范围。"""
    resume_path = getattr(args, "resume_from", None)
    if resume_path:
        network = Network.load_checkpoint(Path(resume_path))
        run_start_tick = network.current_tick + 1
        run_end_tick = network.current_tick + cfg.ticks
        print(
            f"  已从 checkpoint 恢复: {resume_path} (from tick={network.current_tick})"
        )
    else:
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
    return network, run_start_tick, run_end_tick


def _init_logging(
    cfg: SimConfig, args: argparse.Namespace, network: Network
) -> tuple[ObservabilityLogger, NetworkObserver, LiveMonitor | None, int, Path]:
    """初始化日志系统和可视化监控。"""
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
    debug_neuron_ids: list[str] = []
    if network.neurons:
        debug_neuron_ids = [next(iter(network.neurons.keys()))]
    observer = NetworkObserver(network, debug_neuron_ids=debug_neuron_ids)
    logger.log_config(cfg.to_dict())
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
    return logger, observer, live_monitor, checkpoint_interval, checkpoint_dir


def _run_simulation_loop(
    network: Network,
    cfg: SimConfig,
    logger: ObservabilityLogger,
    observer: NetworkObserver,
    live_monitor: LiveMonitor | None,
    run_start_tick: int,
    run_end_tick: int,
    checkpoint_interval: int,
    checkpoint_dir: Path,
) -> tuple[float, AutoLearner]:
    """执行主仿真循环。"""
    auto_learner = AutoLearner(discovery=RuleDiscovery(min_support=2))
    t0 = time.time()
    total_ticks = run_end_tick - run_start_tick + 1
    for t in range(run_start_tick, run_end_tick + 1):
        snapshot = network.tick(t, dt=cfg.dt)
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
                f"  tick {local_progress_tick}/{total_ticks} (global={t}): "
                f"alive={alive}, PE={pe:.4f}, rules={rules}, acc={acc:.3f}"
            )
    return time.time() - t0, auto_learner


def _print_summary(
    *,
    elapsed: float,
    output_path: Path,
    summary: EngineSummary,
    learn_summary: dict[str, Any],
    low_activation_threshold: int,
) -> None:
    """打印仿真结果摘要。"""
    total_spikes = int(summary.get("total_spikes", 0))
    total_propagations = int(summary.get("total_propagations", 0))
    print(f"\n完成! 用时 {elapsed:.2f}s")
    print(f"  输出: {output_path}")
    print(f"  存活: {summary['alive_neurons']}/{summary['total_neurons']}")
    print(f"  总放电: {total_spikes}, 总传播: {total_propagations}")
    print(f"  出生: {summary['birth_count']}, 死亡: {summary['death_count']}")
    print(
        f"  规则数: {learn_summary['rule_count']}, "
        f"预测准确率: {learn_summary['prediction_accuracy']:.3f}"
    )
    print(
        f"  预测次数: {learn_summary['total_predictions']}, "
        f"正确: {learn_summary['correct_predictions']}"
    )
    if total_spikes <= low_activation_threshold or total_propagations <= low_activation_threshold:
        print(
            f"  警告: 低激活（阈值={low_activation_threshold}），"
            f"可提高 --input-weight 或调整 [neuron]/[network] 参数"
        )


def _finalize_run(
    *,
    run_end_tick: int,
    network: Network,
    observer: NetworkObserver,
    logger: ObservabilityLogger,
) -> tuple[EngineSummary, Path]:
    """收尾运行，写出摘要并关闭日志。"""
    final_metrics = observer.get_network_metrics(run_end_tick)
    summary = network.get_summary()
    logger.log_summary(run_end_tick, final_metrics)
    output_path = logger.get_output_path()
    logger.close()
    return summary, output_path


def cmd_run(args: argparse.Namespace) -> None:
    """运行仿真实验。"""
    cfg = build_sim_config(args)
    validate_run_config(cfg)
    seed_all(cfg.seed)
    print(f"[NeuroGenesis Lab v{__version__}] 运行仿真")
    print(f"  ticks={cfg.ticks}, seed={cfg.seed}, schema={SCHEMA_VERSION}")
    network, run_start_tick, run_end_tick = _init_network(cfg, args)
    logger, observer, live_monitor, checkpoint_interval, checkpoint_dir = _init_logging(
        cfg, args, network
    )
    elapsed, auto_learner = _run_simulation_loop(
        network,
        cfg,
        logger,
        observer,
        live_monitor,
        run_start_tick,
        run_end_tick,
        checkpoint_interval,
        checkpoint_dir,
    )
    if live_monitor is not None:
        live_monitor.finalize()
    summary, output_path = _finalize_run(
        run_end_tick=run_end_tick,
        network=network,
        observer=observer,
        logger=logger,
    )
    _print_summary(
        elapsed=elapsed,
        output_path=output_path,
        summary=summary,
        learn_summary=auto_learner.get_summary(),
        low_activation_threshold=max(0, int(cfg.network.low_activation_threshold)),
    )
