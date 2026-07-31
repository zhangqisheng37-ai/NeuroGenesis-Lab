"""单元测试 — 实时可视化服务器（控制状态、WebSocket 端点、广播、CLI 集成）"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

import neurogenesis_lab.cli.run as cli_run
import neurogenesis_lab.server as server_mod
from neurogenesis_lab.live_protocol import (
    MSG_INIT,
    MSG_STATUS,
    MSG_TICK,
    STATE_FINISHED,
)
from neurogenesis_lab.server import (
    LiveControlState,
    LiveServer,
    LiveSession,
    load_frontend_html,
)


def _wait_until(cond: Callable[[], bool], timeout: float = 2.0) -> bool:
    """轮询等待条件成立。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


# ── LiveControlState ────────────────────────────────────────


def test_control_state_pause_resume_stop() -> None:
    control = LiveControlState()
    assert not control.paused and not control.stopped
    control.pause()
    assert control.paused
    control.resume()
    assert not control.paused
    control.stop()
    assert control.stopped


def test_control_state_set_speed_clamps() -> None:
    control = LiveControlState(tick_ms=50)
    control.set_speed(200)
    assert control.speed_ms == 200
    control.set_speed(-5)
    assert control.speed_ms == 0
    control.set_speed("bad")
    assert control.speed_ms == 0


def test_control_state_apply_command() -> None:
    control = LiveControlState()
    control.apply_command("pause", {})
    assert control.paused
    control.apply_command("resume", {})
    assert not control.paused
    control.apply_command("speed_change", {"tick_ms": 120})
    assert control.speed_ms == 120
    control.apply_command("stop", {})
    assert control.stopped


# ── 前端页面 ────────────────────────────────────────────────


def test_load_frontend_html_returns_page() -> None:
    html = load_frontend_html()
    assert "NeuroGenesis" in html
    assert "/ws" in html


def test_index_route_serves_html() -> None:
    server = LiveServer()
    with TestClient(server.app) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "NeuroGenesis" in resp.text


# ── WebSocket 端点 ──────────────────────────────────────────


def test_websocket_sends_init_and_status() -> None:
    server = LiveServer(init_message={"type": MSG_INIT, "total_neurons": 4})
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            init = ws.receive_json()
            status = ws.receive_json()
    assert init["type"] == MSG_INIT
    assert init["total_neurons"] == 4
    assert status["type"] == MSG_STATUS


def test_websocket_pause_command_updates_control() -> None:
    server = LiveServer()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # status（无 init_message）
            ws.send_json({"type": "pause"})
            assert _wait_until(lambda: server.control.paused)


def test_websocket_speed_and_stop_commands() -> None:
    server = LiveServer()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "speed_change", "tick_ms": 150})
            assert _wait_until(lambda: server.control.speed_ms == 150)
            ws.send_json({"type": "stop"})
            assert _wait_until(lambda: server.control.stopped)


def test_websocket_invalid_command_ignored() -> None:
    server = LiveServer()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "garbage"})
            ws.send_json({"type": "pause"})  # 后续合法命令仍应生效
            assert _wait_until(lambda: server.control.paused)


# ── 广播 ────────────────────────────────────────────────────


def test_push_broadcasts_to_client() -> None:
    server = LiveServer()
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # status
            server.push({"type": MSG_TICK, "tick": 7, "neurons": [], "connections": []})
            msg = ws.receive_json()
    assert msg["type"] == MSG_TICK
    assert msg["tick"] == 7


def test_push_without_loop_is_noop() -> None:
    # 未启动服务器（_loop 为 None），push 应静默返回而不抛异常
    server = LiveServer()
    server.push({"type": MSG_TICK, "tick": 1})
    assert server.client_count == 0


def test_server_url_property() -> None:
    server = LiveServer(host="0.0.0.0", port=9000)
    assert server.url == "http://0.0.0.0:9000"


# ── LiveSession ─────────────────────────────────────────────


class _FakeNeuron:
    def __init__(self, nid: str) -> None:
        self.id = nid
        self.pos = (0.0, 0.0)
        self.energy = 1.0
        self.alive = True
        self.spike_count = 0
        self.links: dict[str, dict[str, float]] = {}


class _FakeNetwork:
    def __init__(self) -> None:
        self.neurons = {"a": _FakeNeuron("a")}


def test_live_session_push_and_finalize() -> None:
    server = LiveServer()
    session = LiveSession(server, server.control)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # status
            session.push_tick(1, {"tick": 1, "alive_neurons": 1}, _FakeNetwork())
            tick_msg = ws.receive_json()
            session.finalize(1)
            status_msg = ws.receive_json()
    assert tick_msg["type"] == MSG_TICK
    assert status_msg["type"] == MSG_STATUS
    assert status_msg["state"] == STATE_FINISHED


def test_live_session_pace_zero_is_fast() -> None:
    server = LiveServer(control=LiveControlState(tick_ms=0))
    session = LiveSession(server, server.control)
    start = time.time()
    session.pace()
    assert time.time() - start < 0.5


def test_live_session_wait_while_paused_returns_on_stop() -> None:
    control = LiveControlState()
    control.pause()
    control.stop()  # 已停止：即使暂停也应立即返回
    session = LiveSession(LiveServer(control=control), control)
    start = time.time()
    session.wait_while_paused()
    assert time.time() - start < 0.5


def test_live_session_close_is_safe_without_start() -> None:
    server = LiveServer()
    LiveSession(server, server.control).close()  # 未启动，stop 应安全无异常


# ── 真实 uvicorn 启动/停止（网络受限时跳过）───────────────────


def test_live_server_real_start_and_stop() -> None:
    import httpx

    server = LiveServer(port=8791)
    try:
        server.start(startup_timeout=8.0)
    except Exception as exc:  # 沙箱可能禁止绑定端口
        pytest.skip(f"无法绑定端口: {exc}")
    try:
        resp = httpx.get(server.url + "/", timeout=5.0)
        assert resp.status_code == 200
        assert "NeuroGenesis" in resp.text
    finally:
        server.stop()


# ── run --live-monitor 集成 ─────────────────────────────────


def _run_args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "ticks": 3,
        "seed": 42,
        "dt": 1.0,
        "neurons": 4,
        "expr": "1+1=2",
        "input_weight": 60.0,
        "stim_interval": 1,
        "output": str(tmp_path),
        "experiment_name": None,
        "config": None,
        "debug": False,
        "log_interval": 1,
        "gzip_output": False,
        "max_file_size": 0,
        "async_log": False,
        "log_buffer_size": 256,
        "visualize": False,
        "plot_interval": 5,
        "checkpoint_interval": 0,
        "checkpoint_dir": str(tmp_path),
        "resume_from": None,
        "background_current": None,
        "spontaneous_spike_rate": None,
        "low_activation_threshold": None,
        # live 参数
        "live_monitor": True,
        "host": "127.0.0.1",
        "port": 8793,
        "ui_port": None,
        "live_tick_ms": 0,
        "no_browser": True,
        "wait_for_client": False,
        "client_wait_timeout": 1.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_run_live_monitor_integration(tmp_path: Path, monkeypatch, capsys) -> None:
    # 桩化服务器启动，避免真实绑定端口；push 因 _loop 为 None 成为空操作
    monkeypatch.setattr(server_mod.LiveServer, "start", lambda self, **kwargs: None)
    cli_run.cmd_run(_run_args(tmp_path))
    out = capsys.readouterr().out
    assert "实时监控已启动" in out
    assert "完成!" in out
    assert list(tmp_path.glob("run_*/run.jsonl"))


def test_cmd_run_live_monitor_stops_early(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(server_mod.LiveServer, "start", lambda self, **kwargs: None)

    # 构造一个启动后立即请求停止的会话
    real_init = cli_run.init_live_session

    def stopping_init(cfg: Any, args: Any, network: Any) -> Any:
        session = real_init(cfg, args, network)
        if session is not None:
            session.control.stop()
        return session

    monkeypatch.setattr(cli_run, "init_live_session", stopping_init)
    cli_run.cmd_run(_run_args(tmp_path, ticks=50))
    out = capsys.readouterr().out
    assert "完成!" in out  # 提前停止后仍正常收尾


def test_init_live_session_disabled_returns_none(tmp_path: Path) -> None:
    from neurogenesis_lab.config import SimConfig

    args = _run_args(tmp_path, live_monitor=False)
    session = cli_run.init_live_session(SimConfig(ticks=3), args, _FakeNetwork())
    assert session is None


# ── serve 子命令 ────────────────────────────────────────────


def _serve_args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "ticks": 3,
        "seed": 42,
        "dt": 1.0,
        "neurons": 4,
        "expr": "1+1=2",
        "input_weight": 60.0,
        "stim_interval": 1,
        "config": None,
        "live_monitor": True,
        "host": "127.0.0.1",
        "port": 8795,
        "ui_port": None,
        "live_tick_ms": 0,
        "no_browser": True,
        "wait_for_client": False,
        "client_wait_timeout": 1.0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_serve_streams_then_holds_until_interrupt(tmp_path: Path, monkeypatch, capsys) -> None:
    import neurogenesis_lab.cli.serve as cli_serve

    monkeypatch.setattr(server_mod.LiveServer, "start", lambda self, **kwargs: None)
    # 保活循环首次 sleep 即抛 KeyboardInterrupt，模拟用户 Ctrl+C
    monkeypatch.setattr(
        cli_serve.time, "sleep", lambda _s: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    cli_serve.cmd_serve(_serve_args(tmp_path))
    out = capsys.readouterr().out
    assert "实时服务模式" in out
    assert "服务器保持运行" in out
    assert "收到退出信号" in out


def test_cmd_serve_reports_when_unavailable(tmp_path: Path, monkeypatch, capsys) -> None:
    import neurogenesis_lab.cli.serve as cli_serve

    monkeypatch.setattr(cli_serve, "init_live_session", lambda cfg, args, net: None)
    cli_serve.cmd_serve(_serve_args(tmp_path))
    out = capsys.readouterr().out
    assert "实时服务不可用" in out


def test_main_dispatches_serve(monkeypatch) -> None:
    import sys

    import neurogenesis_lab.cli.parser as cli_parser

    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli_parser, "cmd_serve", lambda args: captured.update(port=args.port))
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "serve", "--port", "8801"])
    cli_parser.main()
    assert captured["port"] == 8801
