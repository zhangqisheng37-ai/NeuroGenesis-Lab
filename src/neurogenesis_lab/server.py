"""实时可视化 WebSocket 服务器 (Phase 8 / P0.2)

基于 FastAPI + uvicorn 实现，提供：
- ``GET /`` 返回自包含前端页面（``web/live.html``）；
- ``WS /ws`` 双向 WebSocket：服务端推送 tick 数据，客户端回传控制命令；
- 线程安全的跨线程广播（仿真主线程 → 服务端事件循环）。

设计要点：uvicorn 在后台守护线程中运行独立事件循环，仿真线程通过
``asyncio.run_coroutine_threadsafe`` 将消息投递到该循环广播给所有客户端。
P5 Web 管理界面复用同一 FastAPI 应用与 WebSocket 协议扩展 REST 端点。

本模块依赖可选组 ``web``（fastapi、uvicorn），未安装时导入将失败——
调用方（``cli/run.py`` / ``cli/serve.py``）以延迟导入方式使用并降级处理。
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from neurogenesis_lab.live_protocol import (
    CMD_PAUSE,
    CMD_RESUME,
    CMD_SPEED,
    CMD_STOP,
    STATE_FINISHED,
    STATE_PAUSED,
    STATE_RUNNING,
    TickMessageBuilder,
    build_status_message,
    clamp_tick_ms,
    parse_command,
)

_STATIC_DIR = Path(__file__).parent / "web"
_FALLBACK_HTML = (
    "<!doctype html><html lang=zh><head><meta charset=utf-8>"
    "<title>NeuroGenesis Live</title></head><body>"
    "<h1>NeuroGenesis Lab 实时监控</h1>"
    "<p>前端页面缺失（web/live.html）。WebSocket 端点仍可用于自定义客户端。</p>"
    "</body></html>"
)


def load_frontend_html() -> str:
    """加载自包含前端页面，缺失时返回兜底 HTML。"""
    path = _STATIC_DIR / "live.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return _FALLBACK_HTML


@dataclass
class LiveControlState:
    """线程安全的仿真控制状态 — 服务端写入，仿真线程读取。

    Attributes:
        tick_ms: 每 tick 间隔（毫秒），用于调速。
    """

    tick_ms: int = 50
    _paused: bool = field(default=False, repr=False)
    _stopped: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def pause(self) -> None:
        """暂停仿真。"""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """继续仿真。"""
        with self._lock:
            self._paused = False

    def stop(self) -> None:
        """请求停止仿真。"""
        with self._lock:
            self._stopped = True

    def set_speed(self, tick_ms: Any) -> None:
        """设置每 tick 间隔（毫秒），自动裁剪到合法区间。"""
        with self._lock:
            self.tick_ms = clamp_tick_ms(tick_ms)

    @property
    def paused(self) -> bool:
        """当前是否暂停。"""
        with self._lock:
            return self._paused

    @property
    def stopped(self) -> bool:
        """当前是否已请求停止。"""
        with self._lock:
            return self._stopped

    @property
    def speed_ms(self) -> int:
        """当前每 tick 间隔（毫秒）。"""
        with self._lock:
            return self.tick_ms

    def apply_command(self, command: str, payload: dict[str, Any]) -> None:
        """应用一条已解析的控制命令。"""
        if command == CMD_PAUSE:
            self.pause()
        elif command == CMD_RESUME:
            self.resume()
        elif command == CMD_STOP:
            self.stop()
        elif command == CMD_SPEED:
            self.set_speed(payload.get("tick_ms"))


class LiveServer:
    """WebSocket 实时可视化服务器 — 封装 FastAPI 应用与后台 uvicorn 线程。"""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        control: LiveControlState | None = None,
        init_message: dict[str, Any] | None = None,
        html: str | None = None,
    ) -> None:
        """构建服务器（尚未启动监听）。

        Args:
            host: 绑定主机。
            port: 绑定端口。
            control: 共享控制状态，None 时新建。
            init_message: 客户端连接后首先收到的初始元数据消息。
            html: 前端页面内容，None 时从 web/live.html 加载。
        """
        self.host = host
        self.port = port
        self.control = control or LiveControlState()
        self._init_message = init_message or {}
        self._html = html if html is not None else load_frontend_html()
        self._clients: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._current_tick = 0
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        """构建 FastAPI 应用（路由 + 生命周期）。"""

        @asynccontextmanager
        async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
            self._loop = asyncio.get_running_loop()
            self._ready.set()
            yield

        app = FastAPI(title="NeuroGenesis Lab Live", lifespan=lifespan)

        @app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(self._html)

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()
            self._clients.add(websocket)
            try:
                if self._init_message:
                    await websocket.send_json(self._init_message)
                await websocket.send_json(self._current_status_message())
                while True:
                    raw = await websocket.receive_json()
                    self._apply_command(raw)
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(websocket)

        return app

    def _current_status_message(self) -> dict[str, Any]:
        """生成反映当前控制状态的 status 消息。"""
        state = STATE_PAUSED if self.control.paused else STATE_RUNNING
        return build_status_message(state, self._current_tick, self.control.speed_ms)

    def _apply_command(self, raw: Any) -> None:
        """解析并应用客户端命令，非法命令静默忽略。"""
        try:
            command, payload = parse_command(raw)
        except ValueError:
            return
        self.control.apply_command(command, payload)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        """向所有客户端广播消息，清理已断开连接。"""
        dead: list[WebSocket] = []
        for websocket in list(self._clients):
            try:
                await websocket.send_json(message)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self._clients.discard(websocket)

    def push(self, message: dict[str, Any], timeout: float = 2.0) -> None:
        """从任意线程推送消息到服务端事件循环并广播（无客户端时快速返回）。

        Args:
            message: 待广播的消息字典。
            timeout: 等待广播完成的最长秒数。
        """
        loop = self._loop
        if loop is None:
            return
        if isinstance(message.get("tick"), int):
            self._current_tick = message["tick"]
        future = asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)
        try:
            future.result(timeout=timeout)
        except Exception:
            pass

    @property
    def client_count(self) -> int:
        """当前已连接客户端数量。"""
        return len(self._clients)

    @property
    def url(self) -> str:
        """服务器访问地址。"""
        return f"http://{self.host}:{self.port}"

    def start(self, startup_timeout: float = 10.0) -> None:
        """在后台守护线程启动 uvicorn，阻塞至就绪。

        Args:
            startup_timeout: 等待启动就绪的最长秒数。

        Raises:
            RuntimeError: 启动超时。
        """
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=startup_timeout):
            raise RuntimeError("Live 服务器启动超时")

    def wait_for_client(self, timeout: float = 30.0) -> bool:
        """阻塞等待至少一个客户端连接。

        Args:
            timeout: 最长等待秒数。

        Returns:
            是否在超时前等到客户端。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._clients:
                return True
            time.sleep(0.05)
        return bool(self._clients)

    def stop(self) -> None:
        """停止服务器并回收后台线程。"""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class LiveSession:
    """实时可视化会话 — 绑定服务器、控制状态与消息构建器，服务于仿真循环。"""

    def __init__(
        self,
        server: LiveServer,
        control: LiveControlState,
        builder: TickMessageBuilder | None = None,
    ) -> None:
        """初始化会话。

        Args:
            server: 已启动的 LiveServer。
            control: 共享控制状态。
            builder: tick 消息构建器，None 时新建默认实例。
        """
        self.server = server
        self.control = control
        self.builder = builder or TickMessageBuilder()

    def push_tick(self, tick: int, snapshot: Any, network: Any) -> None:
        """构建并推送一个 tick 的可视化数据。"""
        self.server.push(self.builder.build(tick, snapshot, network))

    def wait_while_paused(self, poll_interval: float = 0.05) -> None:
        """在暂停期间阻塞，直到继续或收到停止请求。"""
        while self.control.paused and not self.control.stopped:
            time.sleep(poll_interval)

    def pace(self) -> None:
        """按当前调速间隔休眠（speed_ms 毫秒）。"""
        delay = self.control.speed_ms / 1000.0
        if delay > 0:
            time.sleep(delay)

    def finalize(self, tick: int) -> None:
        """推送仿真结束状态。"""
        self.server.push(build_status_message(STATE_FINISHED, tick, self.control.speed_ms))

    def close(self) -> None:
        """关闭底层服务器。"""
        self.server.stop()
