"""CLI web 命令 — 启动 Web 管理界面"""

from __future__ import annotations

import argparse


def cmd_web(args: argparse.Namespace) -> None:
    """启动 Web 管理界面服务器。

    Args:
        args: 已解析的命令行参数（host / port / db）。
    """
    import uvicorn

    from neurogenesis_lab.web.app import create_web_app

    host: str = getattr(args, "host", "127.0.0.1")
    port: int = getattr(args, "port", 8080)
    db: str = getattr(args, "db", "outputs/experiments.db")

    app = create_web_app(db_path=db)
    uvicorn.run(app, host=host, port=port)
