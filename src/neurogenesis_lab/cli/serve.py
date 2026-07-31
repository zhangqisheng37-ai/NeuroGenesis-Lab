"""CLI serve 命令 — 独立实时可视化服务模式 (Phase 8 / P0.2)

启动 WebSocket 服务器并流式推送一次仿真过程；仿真结束后保持服务器运行，
供浏览器持续查看，直至用户 Ctrl+C 退出。与 ``run --live-monitor`` 的区别：
serve 面向"专用观察服务"，不写实验日志与元数据，聚焦实时可视化本身。
"""

from __future__ import annotations

import argparse
import time

from neurogenesis_lab import __version__
from neurogenesis_lab.cli.config_builder import build_sim_config, validate_run_config
from neurogenesis_lab.cli.run import _init_network, init_live_session
from neurogenesis_lab.config import seed_all
from neurogenesis_lab.learning import AutoLearner, RuleDiscovery


def cmd_serve(args: argparse.Namespace) -> None:
    """启动实时可视化服务并流式推送仿真。"""
    # serve 模式默认即开启实时监控
    args.live_monitor = True
    cfg = build_sim_config(args)
    validate_run_config(cfg)
    seed_all(cfg.seed)
    print(f"[NeuroGenesis Lab v{__version__}] 实时服务模式")
    print(f"  ticks={cfg.ticks}, seed={cfg.seed}")
    network, run_start_tick, run_end_tick = _init_network(cfg, args)
    session = init_live_session(cfg, args, network)
    if session is None:
        print("实时服务不可用：请安装 web 依赖 (pip install 'neurogenesis-lab[web]')")
        return

    auto_learner = AutoLearner(discovery=RuleDiscovery(min_support=2))
    last_tick = run_start_tick
    try:
        for t in range(run_start_tick, run_end_tick + 1):
            session.wait_while_paused()
            if session.control.stopped:
                break
            snapshot = network.tick(t, dt=cfg.dt)
            report = auto_learner.on_tick(t, snapshot, network)
            snapshot["rule_count"] = report["rule_count"]
            snapshot["prediction_accuracy"] = report["prediction_accuracy"]
            session.push_tick(t, snapshot, network)
            session.pace()
            last_tick = t
        session.finalize(last_tick)
        print(f"  仿真结束（tick={last_tick}）。服务器保持运行，按 Ctrl+C 退出。")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  收到退出信号，关闭服务器。")
    finally:
        session.close()
