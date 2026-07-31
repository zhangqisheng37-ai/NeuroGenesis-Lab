"""实时监控 REST API 路由 (Phase 4 / P5.3)

提供 Web 管理界面中的实时监控功能：
- ``GET /monitor`` 渲染监控页面（实验运行工作台）
- ``POST /api/run`` 启动后台仿真（支持完整配置，运行结束自动写入实验库）
- ``GET /api/run/status`` 查询仿真状态
- ``POST /api/run/command`` 发送控制命令
- ``WS /ws`` WebSocket 实时数据推送
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from neurogenesis_lab.cli.run import (
    _finalize_run,
    _init_logging,
    _init_network,
    _start_tracking,
)
from neurogenesis_lab.config import SimConfig, seed_all, sim_config_from_dict
from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore, utc_now_iso
from neurogenesis_lab.learning import AutoLearner, RuleDiscovery
from neurogenesis_lab.live_protocol import (
    STATE_FINISHED,
    TickMessageBuilder,
    build_init_message,
    build_status_message,
    clamp_tick_ms,
    parse_command,
)
from neurogenesis_lab.network import Network
from neurogenesis_lab.observability import NetworkObserver, ObservabilityLogger
from neurogenesis_lab.server import LiveControlState

router = APIRouter(tags=["monitor"])
_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

# ── 模块级仿真状态（单实例）───────────────────────────────────
_simulation_lock = threading.Lock()
_simulation_state: dict[str, Any] = {
    "running": False,
    "control": None,      # LiveControlState
    "thread": None,       # threading.Thread
    "config": None,       # SimConfig
    "tick": 0,
    "state": "idle",      # idle / running / paused / finished
    "tick_ms": 50,
    "experiment_id": None,  # 当前运行对应的实验记录 ID
    "message_queue": None,  # deque
    "message_event": None,  # asyncio.Event (lazy, created in event loop)
}
_ws_clients: set[WebSocket] = set()


def _reset_simulation_state() -> None:
    """重置仿真状态中的动态字段。"""
    _simulation_state["control"] = None
    _simulation_state["thread"] = None
    _simulation_state["config"] = None
    _simulation_state["tick"] = 0
    _simulation_state["state"] = "idle"
    _simulation_state["tick_ms"] = 50
    _simulation_state["experiment_id"] = None
    _simulation_state["message_queue"] = None
    _simulation_state["message_event"] = None


async def _broadcast(message: dict[str, Any]) -> None:
    """向所有已连接的 WebSocket 客户端广播消息。"""
    dead: list[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


def _push_message(message: dict[str, Any]) -> None:
    """从仿真线程推送消息到共享队列（跨线程安全）。"""
    queue: deque | None = _simulation_state.get("message_queue")
    event: asyncio.Event | None = _simulation_state.get("message_event")
    if queue is not None:
        queue.append(message)
    if event is not None:
        try:
            event.set()
        except RuntimeError:
            pass


def _finish_tracking_in_thread(
    db_path: str,
    record: ExperimentRecord | None,
    *,
    status: str,
    runtime_seconds: float | None = None,
    summary: dict[str, Any] | None = None,
    final_metrics: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """在仿真线程内写入实验结束元数据（SQLite 连接须在线程内创建）。"""
    if record is None:
        return
    try:
        with ExperimentStore(db_path) as store:
            record.status = status
            record.finished_at = utc_now_iso()
            record.runtime_seconds = runtime_seconds
            record.summary = dict(summary or {})
            record.final_metrics = dict(final_metrics or {})
            record.error = error
            store.save(record)
    except Exception as exc:
        print(f"  警告: 实验元数据写入失败 ({exc})")


def _prepare_run(
    body: dict[str, Any], db_path: str
) -> tuple[
    SimConfig,
    argparse.Namespace,
    Network,
    ObservabilityLogger,
    NetworkObserver,
    ExperimentRecord | None,
    int,
    int,
    str,
]:
    """在主线程中准备一次仿真运行（配置/网络/日志/实验追踪）。

    Args:
        body: 请求体参数（``config`` 优先，其余为基本参数）。
        db_path: 实验数据库路径。

    Returns:
        (cfg, sim_args, network, logger, observer, record, start_tick, end_tick, db_path)。

    Raises:
        ValueError: 参数非法时。
    """
    # 清理服务重启后残留的 running 僵尸记录（单实例语义下旧记录必然已中断）
    try:
        with ExperimentStore(db_path) as cleanup_store:
            cleanup_store.mark_stale_running()
    except Exception as exc:
        print(f"  警告: 清理残留运行记录失败 ({exc})")
    config_data = body.get("config")
    if isinstance(config_data, dict) and config_data:
        cfg = sim_config_from_dict(config_data)
    else:
        cfg = SimConfig()
        cfg.ticks = max(1, int(body.get("ticks", cfg.ticks)))
        cfg.seed = int(body.get("seed", cfg.seed))
        cfg.dt = max(0.0001, float(body.get("dt", cfg.dt)))

    errors = cfg.validate()
    if errors:
        raise ValueError("; ".join(errors))

    sim_args = argparse.Namespace(
        resume_from=None,
        neurons=max(1, int(body.get("neurons", 8))),
        expr=str(body.get("expr", "1+2=3") or "1+2=3"),
        input_weight=max(0.0, float(body.get("input_weight", 40.0) or 40.0)),
        stim_interval=max(1, int(body.get("stim_interval", 5) or 5)),
        output=str(body.get("output", "outputs") or "outputs"),
        debug=False,
        log_interval=max(1, int(body.get("log_interval", 1) or 1)),
        max_file_size=0,
        gzip_output=False,
        async_log=False,
        log_buffer_size=256,
        visualize=False,
        plot_interval=5,
        checkpoint_interval=0,
        checkpoint_dir="outputs",
        experiment_name=body.get("experiment_name"),
        live_monitor=False,
    )

    seed_all(cfg.seed)
    network, start_tick, end_tick = _init_network(cfg, sim_args)
    logger, observer, _live_monitor, _cp_interval, _cp_dir = _init_logging(cfg, sim_args, network)
    store, record = _start_tracking(cfg, sim_args, logger.get_output_path(), db_path=db_path)
    if store is not None:
        # 主线程创建、主线程使用并立即关闭；仿真线程收尾时在自身线程内重建连接
        store.close()
    return (
        cfg,
        sim_args,
        network,
        logger,
        observer,
        record,
        start_tick,
        end_tick,
        db_path,
    )


def _run_simulation_thread(
    *,
    cfg: SimConfig,
    args: argparse.Namespace,
    network: Network,
    logger: ObservabilityLogger,
    observer: NetworkObserver,
    record: ExperimentRecord | None,
    start_tick: int,
    end_tick: int,
    db_path: str,
) -> None:
    """在后台线程中运行仿真循环，结束后写入实验库并推送 finished 状态。"""
    control: LiveControlState = _simulation_state["control"]
    builder = TickMessageBuilder()
    _push_message(build_init_message(cfg.to_dict(), len(network.neurons)))

    auto_learner = AutoLearner(discovery=RuleDiscovery(min_support=2))
    last_tick = start_tick
    t0 = time.time()
    error: str | None = None
    try:
        for t in range(start_tick, end_tick + 1):
            while control.paused and not control.stopped:
                time.sleep(0.05)
            if control.stopped:
                break

            snapshot = network.tick(t, dt=cfg.dt)
            report = auto_learner.on_tick(t, snapshot, network)
            snapshot["rule_count"] = report["rule_count"]
            snapshot["prediction_accuracy"] = report["prediction_accuracy"]

            _push_message(builder.build(t, snapshot, network))
            _simulation_state["tick"] = t
            logger.log_tick(observer, t, snapshot)

            delay = control.speed_ms / 1000.0
            if delay > 0:
                time.sleep(delay)
            last_tick = t
    except Exception as exc:
        error = str(exc) or type(exc).__name__

    elapsed = time.time() - t0
    status = "completed"
    summary_dict: dict[str, Any] = {}
    final_metrics: dict[str, Any] = {}
    try:
        if error is None:
            summary, final_metrics, _output_path = _finalize_run(
                run_end_tick=last_tick,
                network=network,
                observer=observer,
                logger=logger,
            )
            summary_dict = dict(summary)
        else:
            status = "failed"
            logger.close()
    except Exception as exc:
        status = "failed"
        error = str(exc) or type(exc).__name__
        try:
            logger.close()
        except Exception:
            pass
    _finish_tracking_in_thread(
        db_path,
        record,
        status=status,
        runtime_seconds=elapsed,
        summary=summary_dict,
        final_metrics=final_metrics,
        error=error,
    )

    _simulation_state["state"] = "finished"
    _simulation_state["tick_ms"] = control.speed_ms
    # 线程收尾即释放运行锁，避免无 WebSocket 客户端时 running 永远卡住
    _simulation_state["running"] = False
    _push_message(build_status_message(STATE_FINISHED, last_tick, control.speed_ms))


# ── 页面路由 ──────────────────────────────────────────────────


@router.get("/monitor", response_class=HTMLResponse)
async def monitor_page(request: Request) -> HTMLResponse:
    """渲染实时监控页面。"""
    from neurogenesis_lab import __version__

    return _templates.TemplateResponse(
        request,
        "monitor.html",
        {"version": __version__},
    )


# ── REST API ──────────────────────────────────────────────────


@router.post("/api/run")
async def start_simulation(request: Request) -> JSONResponse:
    """启动仿真。

    请求体（可选 JSON）:
        config (dict): 完整配置字典（asdict 或分组格式），优先于基本参数。
        ticks / seed / dt / neurons / expr / input_weight / stim_interval: 基本参数。
        tick_ms (int): 每 tick 间隔毫秒数，默认 50。
        experiment_name (str): 实验名称。

    Returns:
        ``{"status": "started", "experiment_id": "...", "ws_url": "/ws"}`` 或 409/400 错误。
    """
    with _simulation_lock:
        if _simulation_state["running"]:
            return JSONResponse(
                {"error": "仿真已在运行中"},
                status_code=409,
            )

        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}

        try:
            prepared = _prepare_run(body, request.app.state.db_path)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"启动失败: {exc}"}, status_code=500)

        (
            cfg,
            sim_args,
            network,
            logger,
            observer,
            record,
            start_tick,
            end_tick,
            db_path,
        ) = prepared
        tick_ms = clamp_tick_ms(body.get("tick_ms", 50))

        control = LiveControlState(tick_ms=tick_ms)
        message_queue: deque = deque(maxlen=1000)
        _simulation_state["running"] = True
        _simulation_state["control"] = control
        _simulation_state["config"] = cfg
        _simulation_state["tick"] = 0
        _simulation_state["state"] = "running"
        _simulation_state["tick_ms"] = tick_ms
        _simulation_state["experiment_id"] = record.experiment_id if record is not None else None
        _simulation_state["message_queue"] = message_queue
        # message_event 将在 WebSocket 首次连接时创建

        thread = threading.Thread(
            target=_run_simulation_thread,
            kwargs={
                "cfg": cfg,
                "args": sim_args,
                "network": network,
                "logger": logger,
                "observer": observer,
                "record": record,
                "start_tick": start_tick,
                "end_tick": end_tick,
                "db_path": db_path,
            },
            daemon=True,
        )
        _simulation_state["thread"] = thread
        thread.start()

        return JSONResponse(
            {
                "status": "started",
                "experiment_id": _simulation_state["experiment_id"],
                "ws_url": "/ws",
            }
        )


@router.get("/api/run/status")
async def simulation_status() -> JSONResponse:
    """查询当前仿真状态。"""
    control: LiveControlState | None = _simulation_state.get("control")
    if control is not None and _simulation_state["running"]:
        state = _simulation_state["state"]
        if state == "running" and control.paused:
            display_state = "paused"
        else:
            display_state = state
        return JSONResponse(
            {
                "running": True,
                "tick": _simulation_state["tick"],
                "state": display_state,
                "tick_ms": control.speed_ms,
                "experiment_id": _simulation_state.get("experiment_id"),
            }
        )
    return JSONResponse(
        {
            "running": False,
            "tick": 0,
            "state": "idle",
            "tick_ms": 0,
            "experiment_id": None,
        }
    )


@router.post("/api/run/command")
async def send_command(request: Request) -> JSONResponse:
    """发送控制命令到当前仿真。

    请求体: ``{"type": "pause"}`` | ``{"type": "resume"}`` |
            ``{"type": "stop"}`` | ``{"type": "speed_change", "tick_ms": 50}``
    """
    control: LiveControlState | None = _simulation_state.get("control")
    if control is None or not _simulation_state["running"]:
        return JSONResponse({"error": "没有正在运行的仿真"}, status_code=404)

    # 仿真已结束时不再接受命令
    if _simulation_state.get("state") == "finished":
        return JSONResponse({"error": "仿真已结束"}, status_code=404)

    try:
        body = await request.json()
        command, payload = parse_command(body)
    except (ValueError, Exception) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    control.apply_command(command, payload)
    return JSONResponse({"status": "ok", "command": command})


# ── WebSocket ─────────────────────────────────────────────────


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """WebSocket 实时数据推送 — 将仿真线程的消息转发给前端客户端。"""
    await websocket.accept()
    _ws_clients.add(websocket)

    # 确保 message_event 已创建（首次 WebSocket 连接时）
    if _simulation_state.get("message_event") is None:
        _simulation_state["message_event"] = asyncio.Event()

    # 发送当前状态快照
    control: LiveControlState | None = _simulation_state.get("control")
    if control is not None and _simulation_state["running"]:
        state = _simulation_state["state"]
        if state == "running" and control.paused:
            state = "paused"
        await websocket.send_json(
            build_status_message(state, _simulation_state["tick"], control.speed_ms)
        )

    recv_task: asyncio.Task | None = None
    try:
        # 后台任务：接收客户端命令
        async def _receive_loop() -> None:
            try:
                while True:
                    raw = await websocket.receive_json()
                    ctrl: LiveControlState | None = _simulation_state.get("control")
                    if ctrl is not None:
                        try:
                            cmd, payload = parse_command(raw)
                            ctrl.apply_command(cmd, payload)
                        except ValueError:
                            pass
            except (WebSocketDisconnect, Exception):
                pass

        recv_task = asyncio.create_task(_receive_loop())
        event: asyncio.Event = _simulation_state["message_event"]

        while True:
            event.clear()
            # 排空队列中的所有待发消息
            queue: deque | None = _simulation_state.get("message_queue")
            if queue:
                while queue:
                    msg = queue.popleft()
                    await websocket.send_json(msg)

            # 检查仿真是否已结束且队列已排空
            if (
                _simulation_state.get("state") == "finished"
                and _simulation_state.get("thread") is not None
                and not _simulation_state["thread"].is_alive()
            ):
                break

            # 等待新消息（超时 1 秒后重新检查状态）
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _ws_clients.discard(websocket)
        if recv_task is not None:
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass

        # 如果仿真正在结束且我们是最后一个客户端，清理状态
        if (
            _simulation_state.get("state") == "finished"
            and _simulation_state.get("thread") is not None
            and not _simulation_state["thread"].is_alive()
        ):
            _simulation_state["running"] = False
            _reset_simulation_state()
