"""P5 Web 管理界面 — FastAPI 应用工厂

提供仪表盘首页、实验管理 API、配置界面与实时监控端点。
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
    # 全局模板变量：所有页面统一获得版本号（路由无需逐个传入）
    templates.env.globals["version"] = __version__

    # ── 注册路由器 ────────────────────────────────────────────
    from neurogenesis_lab.web.routes import config as config_mod
    from neurogenesis_lab.web.routes import experiments as experiments_mod
    from neurogenesis_lab.web.routes import monitor as monitor_mod

    app.include_router(experiments_mod.router)
    app.include_router(config_mod.router)
    app.include_router(monitor_mod.router)

    # 路由模块各自持有的模板实例同样注入版本号，保持导航/页脚一致
    for mod in (experiments_mod, config_mod, monitor_mod):
        tpls = getattr(mod, "_templates", None) or getattr(mod, "_Templates", None)
        if tpls is not None:
            tpls.env.globals["version"] = __version__

    # ── favicon：避免 404 噪声 ──────────────────────────────

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> JSONResponse:
        return JSONResponse(content=None, status_code=204)


    # ── 仪表盘首页 ────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        """渲染仪表盘首页，从数据库读取真实实验数据。"""
        from neurogenesis_lab.config import default_sim_config
        from neurogenesis_lab.core_bridge import HAS_RUST_CORE
        from neurogenesis_lab.experiments import ExperimentStore

        with ExperimentStore(db_path) as store:
            records = store.list_experiments(limit=5)
            total = len(store.list_experiments())

        cfg = default_sim_config()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "version": __version__,
                "total_experiments": total,
                "recent_experiments": [r.to_dict() for r in records],
                "python_version": sys.version.split()[0],
                "db_path": db_path,
                "rust_core_available": HAS_RUST_CORE,
                "default_ticks": cfg.ticks,
                "default_seed": cfg.seed,
                "default_dt": cfg.dt,
            },
        )

    # ── 配置编辑器页面 ─────────────────────────────────────────

    @app.get("/configure", response_class=HTMLResponse)
    async def configure_page(request: Request) -> HTMLResponse:
        """渲染配置编辑器页面。"""
        return templates.TemplateResponse(request, "config_editor.html", {})

    # ── Benchmark 页面 ──────────────────────────────────────────

    @app.get("/benchmark", response_class=HTMLResponse)
    async def benchmark_page(request: Request) -> HTMLResponse:
        """渲染性能基准测试页面。"""
        return templates.TemplateResponse(
            request,
            "benchmark.html",
            {"version": __version__},
        )

    # ── Benchmark API ───────────────────────────────────────────

    @app.post("/api/benchmark")
    async def run_benchmark(request: Request) -> JSONResponse:
        """运行性能基准测试并返回结果。"""
        import random as _random
        import time as _time

        import numpy as _np

        from neurogenesis_lab.core_bridge import parallel_lif_tick_safe
        from neurogenesis_lab.network import Network
        from neurogenesis_lab.neuron import LIFNeuron
        from neurogenesis_lab.vectorized import VectorizedNetwork

        try:
            body = await request.json()
        except Exception:
            body = {}

        neuron_count = max(1, int(body.get("neurons", 1000)))
        tick_count = max(1, int(body.get("ticks", 100)))
        seed = int(body.get("seed", 42))

        try:
            # 向量化引擎
            _np.random.seed(seed)
            vnet = VectorizedNetwork(max_neurons=neuron_count * 2, min_neurons=2)
            vnet.init_neurons(neuron_count, energy_init=1.5)
            vnet.connect_knn(k=3)
            t0 = _time.time()
            for t in range(1, tick_count + 1):
                vnet.tick(t)
            vec_time = _time.time() - t0

            # 原始引擎（较小规模）
            compare_n = min(neuron_count, 50)
            _random.seed(seed)
            neurons = [LIFNeuron() for _ in range(compare_n)]
            net = Network(
                neurons={n.id: n for n in neurons}, max_neurons=compare_n * 2
            )
            for i, neuron in enumerate(neurons):
                for j in range(1, min(4, compare_n)):
                    target = neurons[(i + j) % compare_n]
                    net.connect(neuron.id, target.id, weight=1.0, delay=1)
            t0 = _time.time()
            for t in range(1, tick_count + 1):
                net.tick(t)
            orig_time = _time.time() - t0

            # Rust 并行 LIF
            rust_size = 10000
            rng = _np.random.default_rng(seed)
            voltages = (
                rng.normal(loc=-65.0, scale=2.0, size=rust_size).astype(float).tolist()
            )
            inputs = (
                rng.normal(loc=0.5, scale=1.0, size=rust_size).astype(float).tolist()
            )
            rust_v = voltages
            rust_spikes = 0
            t0 = _time.time()
            for _ in range(tick_count):
                rust_v, spikes, _backend = parallel_lif_tick_safe(
                    rust_v,
                    v_rest=-65.0,
                    v_reset=-65.0,
                    v_th=-55.0,
                    tau_m=20.0,
                    r_m=1.0,
                    inputs=inputs,
                    dt=1.0,
                )
                rust_spikes += sum(1 for s in spikes if s)
            rust_time = _time.time() - t0

            speedup = orig_time / vec_time if vec_time > 0 and compare_n == neuron_count else None

            return JSONResponse(
                {
                    "vectorized": {
                        "neurons": neuron_count,
                        "time": round(vec_time, 4),
                    },
                    "original": {
                        "neurons": compare_n,
                        "time": round(orig_time, 4),
                    },
                    "rust": {
                        "neurons": rust_size,
                        "time": round(rust_time, 4),
                        "spikes": rust_spikes,
                    },
                    "speedup": round(speedup, 1) if speedup else None,
                }
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    # ── 系统信息 API ───────────────────────────────────────────

    @app.get("/api/info")
    async def system_info() -> JSONResponse:
        """返回系统完整信息（版本、Rust 核心状态、默认配置等）。"""
        from neurogenesis_lab.config import default_sim_config
        from neurogenesis_lab.core_bridge import HAS_RUST_CORE, rust_core_status

        cfg = default_sim_config()
        return JSONResponse(
            {
                "version": __version__,
                "python_version": sys.version.split()[0],
                "rust_core": {
                    "available": HAS_RUST_CORE,
                    "components": rust_core_status(),
                },
                "defaults": {
                    "ticks": cfg.ticks,
                    "seed": cfg.seed,
                    "dt": cfg.dt,
                    "neurons": 8,
                    "schema_version": cfg.schema_version,
                },
            }
        )

    return app
