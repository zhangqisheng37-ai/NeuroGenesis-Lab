"""P5 Web 管理界面 — FastAPI 应用工厂

提供仪表盘首页、实验管理 API、配置界面与实时监控端点。
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from neurogenesis_lab import __version__

# 相对于本文件定位模板与静态文件目录（安装后仍存在）
_BASE_DIR = Path(__file__).parent
_TEMPLATE_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"


def create_web_app(db_path: str = "outputs/experiments.db") -> FastAPI:
    """创建并配置 P5 Web 管理界面 FastAPI 应用。

    Args:
        db_path: 实验数据库路径，供路由处理器读取。

    Returns:
        配置完成的 FastAPI 应用实例。
    """
    app = FastAPI(
        title="NeuroGenesis Lab Web",
        version=__version__,
    )

    # 将数据库路径挂载到 app.state，供路由处理器读取
    app.state.db_path = db_path

    # 静态文件与模板
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

    # ── 注册路由器 ────────────────────────────────────────────
    from neurogenesis_lab.web.routes.config import router as config_router
    from neurogenesis_lab.web.routes.experiments import router as experiments_router
    from neurogenesis_lab.web.routes.monitor import router as monitor_router

    app.include_router(experiments_router)
    app.include_router(config_router)
    app.include_router(monitor_router)

    # ── 仪表盘首页 ────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        """渲染仪表盘首页，从数据库读取真实实验数据。"""
        from neurogenesis_lab.experiments import ExperimentStore

        with ExperimentStore(db_path) as store:
            records = store.list_experiments(limit=5)
            total = len(store.list_experiments())

        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "version": __version__,
                "total_experiments": total,
                "recent_experiments": [r.to_dict() for r in records],
                "python_version": sys.version.split()[0],
                "db_path": db_path,
            },
        )

    return app
