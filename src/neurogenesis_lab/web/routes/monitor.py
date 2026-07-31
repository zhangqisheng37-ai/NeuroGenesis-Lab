"""实时监控 REST API 路由 (Phase 4 / P5.3)

提供 Web 管理界面中的实时监控功能：
- ``GET /monitor`` 渲染监控页面
- ``POST /api/run`` 启动后台仿真
- ``GET /api/run/status`` 查询仿真状态
- ``POST /api/run/command`` 发送控制命令
- ``WS /ws`` WebSocket 实时数据推送
"""

from __future__ import annotations

import argparse
import asyncio
import threading
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from neurogenesis_lab.config import SimConfig, seed_all
from neurogenesis_lab.live_protocol import (
    STATE_FINISHED,
    TickMessageBuilder,
    build_init_message,
    build_status_message,
    parse_command,
)
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


def _run_simulation_thread(cfg: SimConfig, args: argparse.Namespace) -> None:
    """在后台线程中运行仿真循环（参考 serve.py 模式）。"""
    from neurogenesis_lab.cli.run import _init_network
    from neurogenesis_lab.learning import AutoLearner, RuleDiscovery

    try:
        network, start_tick, end_tick = _init_network(cfg, args)
        control: LiveControlState = _simulation_state["control"]
        queue: deque = _simulation_state["message_queue"]
        builder = TickMessageBuilder()
        init_msg = build_init_message(cfg.to_dict(), len(network.neurons))
        queue.append(init_msg)

        auto_learner = AutoLearner(discovery=RuleDiscovery(min_support=2))
        last_tick = start_tick

        for t in range(start_tick, end_tick + 1):
            while control.paused and not control.stopped:
                import time as _time

                _time.sleep(0.05)
            if control.stopped:
                break

            snapshot = network.tick(t, dt=cfg.dt)
            report = auto_learner.on_tick(t, snapshot, network)
            snapshot["rule_count"] = report["rule_count"]
            snapshot["prediction_accuracy"] = report["prediction_accuracy"]

            msg = builder.build(t, snapshot, network)
            queue.append(msg)
            _simulation_state["tick"] = t

            delay = control.speed_ms / 1000.0
            if delay > 0:
                import time as _time

                _time.sleep(delay)
            last_tick = t

        _simulation_state["state"] = "finished"
        _simulation_state["tick_ms"] = control.speed_ms
        queue.append(build_status_message(STATE_FINISHED, last_tick, control.speed_ms))
        # 唤醒可能正在等待的 WebSocket 处理器
        event = _simulation_state.get("message_event")
        if event is not None:
            try:
                event.set()
            except RuntimeError:
                pass
    except Exception:
        _simulation_state["state"] = "finished"
        queue = _simulation_state.get("message_queue")
        if queue is not None:
            queue.append(
                build_status_message(STATE_FINISHED, _simulation_state.get("tick", 0), 50)
            )
        event = _simulation_state.get("message_event")
        if event is not None:
            try:
                event.set()
            except RuntimeError:
                pass


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
        ticks (int): 仿真步数，默认 100。
        seed (int): 随机种子，默认 42。
        neurons (int): 神经元数量，默认 8。

    Returns:
        ``{"status": "started", "ws_url": "/ws"}`` 或 409/400 错误。
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

        ticks = max(1, int(body.get("ticks", 100)))
        seed = int(body.get("seed", 42))
        neurons = max(1, int(body.get("neurons", 8)))

        cfg = SimConfig(ticks=ticks, seed=seed)
        errors = cfg.validate()
        if errors:
            return JSONResponse({"error": "; ".join(errors)}, status_code=400)

        sim_args = argparse.Namespace(
            resume_from=None,
            neurons=neurons,
            expr="1+2=3",
            input_weight=40.0,
            stim_interval=5,
        )

        seed_all(seed)

        control = LiveControlState(tick_ms=50)
        message_queue: deque = deque(maxlen=1000)
        _simulation_state["running"] = True
        _simulation_state["control"] = control
        _simulation_state["config"] = cfg
        _simulation_state["tick"] = 0
        _simulation_state["state"] = "running"
        _simulation_state["tick_ms"] = 50
        _simulation_state["message_queue"] = message_queue
        # message_event 将在 WebSocket 首次连接时创建

        thread = threading.Thread(
            target=_run_simulation_thread,
            args=(cfg, sim_args),
            daemon=True,
        )
        _simulation_state["thread"] = thread
        thread.start()

        return JSONResponse({"status": "started", "ws_url": "/ws"})


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
            }
        )
    return JSONResponse(
        {
            "running": False,
            "tick": 0,
            "state": "idle",
            "tick_ms": 0,
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
