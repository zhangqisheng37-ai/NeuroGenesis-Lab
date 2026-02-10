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
from neurogenesis_lab.config import SCHEMA_VERSION, SimConfig, seed_all
from neurogenesis_lab.encoding import Encoder, Event
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
    if hasattr(args, "config") and args.config:
        toml_data = _load_toml_config(Path(args.config))
        sim = toml_data.get("simulation", {})
        if "ticks" in sim:
            cfg = SimConfig(ticks=sim["ticks"], seed=cfg.seed, dt=cfg.dt)
        if "seed" in sim:
            cfg = SimConfig(ticks=cfg.ticks, seed=sim["seed"], dt=cfg.dt)
    # 命令行覆盖
    if hasattr(args, "ticks") and args.ticks is not None:
        cfg = SimConfig(ticks=args.ticks, seed=cfg.seed, dt=cfg.dt)
    if hasattr(args, "seed") and args.seed is not None:
        cfg = SimConfig(ticks=cfg.ticks, seed=args.seed, dt=cfg.dt)
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

    # 创建网络
    neuron_count = getattr(args, "neurons", 8) or 8
    neuron_types = [LIFNeuron, IzhikevichNeuron, AdExNeuron]
    neurons = [neuron_types[i % len(neuron_types)]() for i in range(neuron_count)]
    network = Network(
        neurons={n.id: n for n in neurons},
        max_neurons=cfg.network.max_neurons,
        max_links=cfg.network.max_links,
        reproduction_threshold=cfg.network.reproduction_threshold,
        mutation_scale=cfg.network.mutation_scale,
    )
    # 环形连接
    for i, neuron in enumerate(neurons):
        target = neurons[(i + 1) % len(neurons)]
        network.connect(neuron.id, target.id, weight=1.0, delay=1)

    # 注入信号
    encoder = Encoder()
    expr = getattr(args, "expr", "1+2=3") or "1+2=3"
    tokens = encoder.encode_expr(expr)
    for tick_start in range(1, min(cfg.ticks, 20), 5):
        base_events = encoder.to_events(tokens, src_id=neurons[0].id)
        scheduled = [
            Event(token_id=e.token_id, t=e.t + tick_start, src_id=e.src_id, meta=e.meta)
            for e in base_events
        ]
        network.enqueue(scheduled)

    # 输出设置
    output_dir = Path(getattr(args, "output", "outputs") or "outputs")
    logger = create_experiment_logger(output_dir, "run", debug_mode=bool(getattr(args, "debug", False)))
    observer = NetworkObserver(network, debug_neuron_ids=[neurons[0].id])
    logger.log_config(cfg.to_dict())

    # 运行
    t0 = time.time()
    for t in range(1, cfg.ticks + 1):
        snapshot = network.tick(t, dt=cfg.dt)
        logger.log_tick(observer, t, snapshot)
        if t % max(1, cfg.ticks // 5) == 0:
            alive = snapshot["alive_neurons"]
            pe = snapshot["prediction_error"]
            print(f"  tick {t}/{cfg.ticks}: alive={alive}, PE={pe:.4f}")

    elapsed = time.time() - t0
    final = observer.get_network_metrics(cfg.ticks)
    summary = network.get_summary()
    logger.log_summary(cfg.ticks, final)

    print(f"\n完成! 用时 {elapsed:.2f}s")
    print(f"  输出: {logger.get_output_path()}")
    print(f"  存活: {summary['alive_neurons']}/{summary['total_neurons']}")
    print(f"  总放电: {summary['total_spikes']}, 总传播: {summary['total_propagations']}")
    print(f"  出生: {summary['birth_count']}, 死亡: {summary['death_count']}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """性能基准测试"""
    import numpy as np
    from neurogenesis_lab.vectorized import VectorizedNetwork

    neuron_count = getattr(args, "neurons", 1000) or 1000
    tick_count = getattr(args, "ticks", 100) or 100
    seed = getattr(args, "seed", 42) or 42

    print(f"[NeuroGenesis Lab] 性能基准测试")
    print(f"  neurons={neuron_count}, ticks={tick_count}, seed={seed}")

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

    print(f"\n结果:")
    print(f"  向量化引擎 ({neuron_count} neurons): {vec_time:.3f}s ({tick_count / vec_time:.0f} ticks/s)")
    print(f"  原始引擎 ({compare_n} neurons): {orig_time:.3f}s ({tick_count / orig_time:.0f} ticks/s)")
    if compare_n == neuron_count and orig_time > 0:
        print(f"  加速比: {orig_time / vec_time:.1f}x")
    print(f"  向量化存活: {vec_summary['alive_neurons']}")


def cmd_info(args: argparse.Namespace) -> None:
    """显示版本与配置信息"""
    print(f"NeuroGenesis Lab v{__version__}")
    print(f"Schema Version: {SCHEMA_VERSION}")
    cfg = SimConfig()
    print(f"默认配置:")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neurogenesis-lab",
        description="NeuroGenesis Lab — 数字生命仿真平台",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # run
    run_parser = subparsers.add_parser("run", help="运行仿真实验")
    run_parser.add_argument("--ticks", type=int, default=None, help="仿真步数")
    run_parser.add_argument("--seed", type=int, default=None, help="随机种子")
    run_parser.add_argument("--neurons", type=int, default=8, help="初始神经元数")
    run_parser.add_argument("--expr", type=str, default="1+2=3", help="输入表达式")
    run_parser.add_argument("--output", type=str, default="outputs", help="输出目录")
    run_parser.add_argument("--config", type=str, default=None, help="TOML 配置文件路径")
    run_parser.add_argument("--debug", action="store_true", help="开启 debug 模式")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="性能基准测试")
    bench_parser.add_argument("--neurons", type=int, default=1000, help="神经元数量")
    bench_parser.add_argument("--ticks", type=int, default=100, help="仿真步数")
    bench_parser.add_argument("--seed", type=int, default=42, help="随机种子")

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
