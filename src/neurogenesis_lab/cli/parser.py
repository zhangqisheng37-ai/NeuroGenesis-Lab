"""CLI 参数解析 — argparse 定义与命令分派"""

from __future__ import annotations

import argparse

from neurogenesis_lab import __version__
from neurogenesis_lab.cli.benchmark import cmd_benchmark
from neurogenesis_lab.cli.compare import cmd_compare
from neurogenesis_lab.cli.delete import cmd_delete
from neurogenesis_lab.cli.info import cmd_info
from neurogenesis_lab.cli.list import cmd_list
from neurogenesis_lab.cli.run import cmd_run
from neurogenesis_lab.cli.serve import cmd_serve
from neurogenesis_lab.cli.web import cmd_web

_DEFAULT_DB = "outputs/experiments.db"


def _add_live_args(parser: argparse.ArgumentParser) -> None:
    """为 run / serve 添加实时可视化（WebSocket）共享参数。"""
    parser.add_argument(
        "--live-monitor",
        action="store_true",
        help="启用实时 WebSocket 可视化监控（浏览器观察）",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="实时服务绑定主机")
    parser.add_argument("--port", type=int, default=8765, help="实时服务端口")
    parser.add_argument(
        "--ui-port",
        type=int,
        default=None,
        help="浏览器打开的 UI 端口（默认同 --port，用于反向代理场景）",
    )
    parser.add_argument(
        "--live-tick-ms",
        type=int,
        default=50,
        help="每 tick 推送间隔（毫秒，用于调速）",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器",
    )
    parser.add_argument(
        "--wait-for-client",
        action="store_true",
        help="启动后等待至少一个客户端连接再开始仿真",
    )
    parser.add_argument(
        "--client-wait-timeout",
        type=float,
        default=30.0,
        help="等待客户端连接的超时秒数",
    )


def _build_parser() -> argparse.ArgumentParser:
    """构建完整的命令行参数解析器。"""
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
        "--experiment-name",
        type=str,
        default=None,
        help="实验名称（记录到实验元数据，便于 list/compare 检索）",
    )
    run_parser.add_argument("--config", type=str, default=None, help="TOML 配置文件路径")
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
    _add_live_args(run_parser)

    # serve — 独立实时可视化服务
    serve_parser = subparsers.add_parser("serve", help="启动实时可视化服务（流式推送仿真）")
    serve_parser.add_argument("--ticks", type=int, default=None, help="仿真步数")
    serve_parser.add_argument("--seed", type=int, default=None, help="随机种子")
    serve_parser.add_argument("--dt", type=float, default=None, help="时间步长")
    serve_parser.add_argument("--neurons", type=int, default=8, help="初始神经元数")
    serve_parser.add_argument("--expr", type=str, default="1+2=3", help="输入表达式")
    serve_parser.add_argument(
        "--input-weight", type=float, default=40.0, help="外部注入事件的默认权重"
    )
    serve_parser.add_argument(
        "--stim-interval", type=int, default=5, help="外部刺激注入间隔（tick）"
    )
    serve_parser.add_argument("--config", type=str, default=None, help="TOML 配置文件路径")
    _add_live_args(serve_parser)

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

    # web — Web 管理界面
    web_parser = subparsers.add_parser("web", help="启动 Web 管理界面")
    web_parser.add_argument("--host", type=str, default="127.0.0.1", help="绑定主机")
    web_parser.add_argument("--port", type=int, default=8080, help="绑定端口")
    web_parser.add_argument("--db", type=str, default=_DEFAULT_DB, help="实验数据库路径")

    # info
    subparsers.add_parser("info", help="显示版本与配置信息")

    # list
    list_parser = subparsers.add_parser("list", help="列出已追踪的实验")
    list_parser.add_argument("--db", type=str, default=_DEFAULT_DB, help="实验数据库路径")
    list_parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help='过滤表达式，如 "neurons>100,seed=42"（op: > < >= <= = !=，多条件用 , 或 and 连接）',
    )
    list_parser.add_argument("--limit", type=int, default=0, help="最多显示条数，0=不限")

    # compare
    compare_parser = subparsers.add_parser("compare", help="对比多个实验的参数与结果")
    compare_parser.add_argument("experiment_ids", nargs="+", help="待对比的实验 ID（≥2 个）")
    compare_parser.add_argument("--db", type=str, default=_DEFAULT_DB, help="实验数据库路径")

    # delete
    delete_parser = subparsers.add_parser("delete", help="删除实验元数据与输出文件")
    delete_parser.add_argument("experiment_ids", nargs="+", help="待删除的实验 ID")
    delete_parser.add_argument("--db", type=str, default=_DEFAULT_DB, help="实验数据库路径")
    delete_parser.add_argument(
        "--keep-files",
        action="store_true",
        help="仅删除元数据，保留输出文件",
    )

    return parser


def main() -> None:
    """CLI 主入口 — 解析参数并分派到对应命令。"""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "delete":
        cmd_delete(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
