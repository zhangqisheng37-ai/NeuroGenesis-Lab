"""Phase 4 监控 API 测试 — 状态查询/启动主流程与错误分支、状态机边界、WebSocket 端点

原 test_web_monitor.py 与 test_web_monitor_branches.py 合并。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore
from neurogenesis_lab.server import LiveControlState
from neurogenesis_lab.web.app import create_web_app
from neurogenesis_lab.web.routes import monitor


@pytest.fixture
def client(tmp_path):
    """创建测试客户端。"""
    db = tmp_path / "test.db"
    app = create_web_app(db_path=str(db))
    return TestClient(app)


def _wait_idle(client: TestClient, timeout: float = 15.0) -> dict:
    """轮询等待仿真结束并复位，返回最终状态。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get("/api/run/status").json()
        if not data["running"]:
            return data
        time.sleep(0.05)
    raise TimeoutError("仿真未在预期时间内结束")


def test_monitor_page_returns_200(client):
    """监控页面返回 200。"""
    resp = client.get("/monitor")
    assert resp.status_code == 200
    assert "实时监控" in resp.text or "monitor" in resp.text.lower()


def test_simulation_status_idle(client):
    """空闲状态查询返回 idle。"""
    data = client.get("/api/run/status").json()
    assert data["running"] is False
    assert data["state"] == "idle"


def test_start_simulation(client, tmp_path):
    """启动仿真返回 started 状态与实验 ID。"""
    resp = client.post(
        "/api/run",
        json={"ticks": 5, "seed": 1, "neurons": 4, "output": str(tmp_path), "tick_ms": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert "ws_url" in data
    assert data["experiment_id"]
    _wait_idle(client)


def test_start_simulation_conflict(client, tmp_path):
    """重复启动仿真返回 409。"""
    client.post(
        "/api/run",
        json={"ticks": 50, "seed": 1, "neurons": 4, "output": str(tmp_path), "tick_ms": 0},
    )
    resp = client.post(
        "/api/run",
        json={"ticks": 50, "seed": 2, "neurons": 4, "output": str(tmp_path), "tick_ms": 0},
    )
    assert resp.status_code == 409
    _wait_idle(client)


def test_simulation_finishes_and_releases_run_lock(client, tmp_path):
    """仿真结束后 running 必须复位：无 WebSocket 客户端也能再次启动，且结果落库。"""
    db = tmp_path / "test.db"
    app = create_web_app(db_path=str(db))
    tc = TestClient(app)

    resp = tc.post(
        "/api/run",
        json={"ticks": 3, "seed": 7, "neurons": 4, "output": str(tmp_path), "tick_ms": 0},
    )
    assert resp.status_code == 200
    experiment_id = resp.json()["experiment_id"]

    # 不连接任何 WebSocket，等待线程结束
    status = _wait_idle(tc)
    assert status["state"] == "idle"

    # 再次启动必须成功（回归：修复前 running 永不复位导致 409）
    resp2 = tc.post(
        "/api/run",
        json={"ticks": 3, "seed": 8, "neurons": 4, "output": str(tmp_path), "tick_ms": 0},
    )
    assert resp2.status_code == 200
    _wait_idle(tc)

    # 实验库中出现 completed 记录
    from neurogenesis_lab.experiments import ExperimentStore

    with ExperimentStore(db) as store:
        record = store.get(experiment_id)
    assert record is not None
    assert record.status == "completed"
    assert record.seed == 7
    assert record.summary
    assert record.final_metrics

    # 输出目录包含 run.jsonl
    jsonl = list((tmp_path / experiment_id).glob("run.jsonl*"))
    assert jsonl, "run.jsonl 输出缺失"


def test_start_simulation_with_full_config(client, tmp_path):
    """携带完整 config（asdict 格式）启动仿真。"""
    resp = client.post(
        "/api/run",
        json={
            "config": {"ticks": 3, "seed": 5, "dt": 1.0, "neuron": {"tau_m": 15.0}},
            "output": str(tmp_path),
            "tick_ms": 0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert data["experiment_id"]
    _wait_idle(client)


def test_start_simulation_invalid_config(client, tmp_path):
    """非法配置（ticks<=0）返回 400 且不启动仿真。"""
    resp = client.post(
        "/api/run",
        json={"config": {"ticks": 0}, "output": str(tmp_path)},
    )
    assert resp.status_code == 400
    assert "ticks" in resp.json()["error"]
    status = client.get("/api/run/status").json()
    assert status["running"] is False


def test_command_no_simulation(client):
    """无仿真时发送命令返回 404 或仿真已结束。"""
    # 模块级仿真状态可能已被前面的测试启动，这里只验证端点可访问
    resp = client.post("/api/run/command", json={"type": "pause"})
    assert resp.status_code in (200, 404)


def test_dashboard_shows_real_data(client, tmp_path):
    """仪表盘从数据库读取真实数据。"""
    from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore

    db_path = str(tmp_path / "real.db")
    with ExperimentStore(db_path) as store:
        store.save(
            ExperimentRecord(
                experiment_id="dash_test_001",
                experiment_name="仪表盘测试",
                status="completed",
                seed=99,
                ticks=50,
                neurons=6,
                runtime_seconds=0.5,
                config={},
            )
        )

    from neurogenesis_lab.web.app import create_web_app

    app = create_web_app(db_path=db_path)
    tc = TestClient(app)
    resp = tc.get("/")
    assert resp.status_code == 200
    assert "dash_test_001" in resp.text
    assert "1" in resp.text  # total_experiments


def test_dashboard_shows_rust_core_and_defaults(client):
    """仪表盘显示 Rust 核心状态和默认配置。"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Rust" in resp.text
    assert "ticks" in resp.text


def test_benchmark_page_returns_200(client):
    """Benchmark 页面返回 200。"""
    resp = client.get("/benchmark")
    assert resp.status_code == 200
    assert "基准测试" in resp.text or "benchmark" in resp.text.lower()


def test_api_info(client):
    """系统信息 API 返回完整信息。"""
    data = client.get("/api/info").json()
    assert "version" in data
    assert "python_version" in data
    assert "rust_core" in data
    assert "available" in data["rust_core"]
    assert "defaults" in data
    assert "ticks" in data["defaults"]
    assert "seed" in data["defaults"]


def test_api_benchmark(client):
    """Benchmark API 返回性能测试结果。"""
    resp = client.post(
        "/api/benchmark",
        json={"neurons": 10, "ticks": 5, "seed": 42},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "vectorized" in data
    assert "original" in data
    assert "rust" in data
    assert "time" in data["vectorized"]
    assert "neurons" in data["vectorized"]


# ======================================================================
# 合并自 test_web_monitor_branches.py：monitor.py 分支覆盖测试
# ======================================================================


# ── 隔离与工具 ────────────────────────────────────────────────


def _clean_state() -> None:
    """把模块级仿真状态恢复到空闲初值。"""
    monitor._reset_simulation_state()
    monitor._simulation_state["running"] = False
    monitor._ws_clients.clear()


@pytest.fixture(autouse=True)
def _isolate_simulation_state() -> Any:
    """每个用例前后重置全局单例状态，避免跨用例污染。"""
    _clean_state()
    yield
    _clean_state()


def _finished_thread() -> threading.Thread:
    """返回一个已结束的线程，用于满足 WebSocket 的退出条件。"""
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    return thread


def _wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> None:
    """轮询等待条件成立（带确定性上限，不使用固定长睡眠）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("条件未在预期时间内满足")


# ── 替身 ─────────────────────────────────────────────────────


class _FakeNeuron:
    """满足 TickMessageBuilder 读取需求的最小神经元替身。"""

    spike_count = 0
    pos = (0.0, 0.0)
    energy = 1.0
    alive = True
    links: dict[str, Any] = {}


class _FakeNetwork:
    """最小网络替身；可在指定 tick 强制抛出异常。"""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.neurons = {"n1": _FakeNeuron()}
        self._fail_at = fail_at

    def tick(self, tick: int, dt: float = 1.0) -> dict[str, Any]:
        if self._fail_at is not None and tick == self._fail_at:
            raise RuntimeError("模拟 tick 失败")
        return {"tick": tick, "alive_neurons": 1, "total_neurons": 1}


class _FakeCfg:
    """最小配置替身。"""

    dt = 1.0
    ticks = 3
    seed = 42

    def to_dict(self) -> dict[str, Any]:  # noqa: D102 - 替身
        return {"ticks": self.ticks, "seed": self.seed}


class _FakeLogger:
    """记录调用序列的日志替身。"""

    def __init__(self) -> None:
        self.logged: list[int] = []
        self.closed = False

    def log_tick(self, observer: Any, tick: int, snapshot: Any) -> None:  # noqa: D102
        self.logged.append(tick)

    def close(self) -> None:  # noqa: D102
        self.closed = True

    def get_output_path(self) -> str:  # noqa: D102
        return "outputs/_fake/run.jsonl"


class _FakeLearner:
    """AutoLearner 替身，避免依赖真实快照结构。"""

    def on_tick(self, tick: int, snapshot: Any, network: Any) -> dict[str, Any]:  # noqa: D102
        return {"rule_count": 0, "prediction_accuracy": 0.0}


class _FakeControl:
    """可控控制状态替身，用于精确命中暂停/停止分支。"""

    def __init__(self, *, paused_once: bool = False, stopped: bool = False, speed_ms: int = 0):
        self._paused_once = paused_once
        self._stopped = stopped
        self._speed_ms = speed_ms
        self._pause_reads = 0

    @property
    def paused(self) -> bool:
        """首次读取为 True（触发一次等待），之后为 False。"""
        self._pause_reads += 1
        return self._paused_once and self._pause_reads == 1

    @property
    def stopped(self) -> bool:
        """是否已请求停止。"""
        return self._stopped

    @property
    def speed_ms(self) -> int:
        """每 tick 间隔毫秒数。"""
        return self._speed_ms


# ── 状态管理与消息推送 ────────────────────────────────────────


def test_reset_simulation_state_clears_dynamic_fields() -> None:
    """重置函数应清空全部动态字段，但保留 running 键交由调用方管理。"""
    monitor._simulation_state.update(
        {
            "control": object(),
            "thread": object(),
            "config": object(),
            "tick": 42,
            "state": "finished",
            "tick_ms": 999,
            "experiment_id": "exp-1",
            "message_queue": deque(),
            "message_event": object(),
        }
    )

    monitor._reset_simulation_state()

    assert monitor._simulation_state["control"] is None
    assert monitor._simulation_state["thread"] is None
    assert monitor._simulation_state["config"] is None
    assert monitor._simulation_state["tick"] == 0
    assert monitor._simulation_state["state"] == "idle"
    assert monitor._simulation_state["tick_ms"] == 50
    assert monitor._simulation_state["experiment_id"] is None
    assert monitor._simulation_state["message_queue"] is None
    assert monitor._simulation_state["message_event"] is None


def test_push_message_appends_and_signals_event() -> None:
    """消息应进入共享队列并唤醒事件，供 WebSocket 协程排空。"""
    queue: deque = deque(maxlen=10)

    class _Event:
        def __init__(self) -> None:
            self.signaled = False

        def set(self) -> None:
            self.signaled = True

    event = _Event()
    monitor._simulation_state["message_queue"] = queue
    monitor._simulation_state["message_event"] = event

    monitor._push_message({"type": "tick_update", "tick": 1})

    assert list(queue) == [{"type": "tick_update", "tick": 1}]
    assert event.signaled is True


def test_push_message_swallows_event_loop_closed_error() -> None:
    """事件循环已关闭时 set() 抛 RuntimeError，必须被吞掉（跨线程安全）。"""

    class _ClosedEvent:
        def set(self) -> None:
            raise RuntimeError("事件循环已关闭")

    monitor._simulation_state["message_queue"] = deque()
    monitor._simulation_state["message_event"] = _ClosedEvent()

    monitor._push_message({"type": "status"})  # 不得抛出


def test_push_message_tolerates_missing_queue_and_event() -> None:
    """队列与事件均未就绪时推送应静默跳过。"""
    monitor._simulation_state["message_queue"] = None
    monitor._simulation_state["message_event"] = None

    monitor._push_message({"type": "init"})  # 不得抛出


# ── 实验元数据收尾 ────────────────────────────────────────────


def test_finish_tracking_noop_without_record(tmp_path: Any) -> None:
    """无实验记录时直接返回，不得创建数据库文件。"""
    db_path = tmp_path / "no_record.db"

    monitor._finish_tracking_in_thread(str(db_path), None, status="completed")

    assert not db_path.exists()


def test_finish_tracking_swallows_store_failure(monkeypatch: Any) -> None:
    """落库异常只打印警告，不得向上抛出（否则仿真线程崩溃）。"""

    class _BrokenStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("数据库已损坏")

    monkeypatch.setattr(monitor, "ExperimentStore", _BrokenStore)

    monitor._finish_tracking_in_thread("unused.db", object(), status="completed")  # 不得抛出


def test_finish_tracking_persists_completed_record(tmp_path: Any) -> None:
    """正常路径应写入状态、耗时、摘要与最终指标。"""
    db_path = str(tmp_path / "track.db")
    record = ExperimentRecord(experiment_id="track_001", status="running")

    monitor._finish_tracking_in_thread(
        db_path,
        record,
        status="completed",
        runtime_seconds=1.25,
        summary={"alive_neurons": 4},
        final_metrics={"tick": 10},
    )

    with ExperimentStore(db_path) as store:
        saved = store.get("track_001")
    assert saved is not None
    assert saved.status == "completed"
    assert saved.runtime_seconds == 1.25
    assert saved.summary == {"alive_neurons": 4}
    assert saved.final_metrics == {"tick": 10}
    assert saved.error is None


# ── 运行准备 ──────────────────────────────────────────────────


def test_prepare_run_tolerates_stale_cleanup_failure(monkeypatch: Any, tmp_path: Any) -> None:
    """清理僵尸记录失败不应阻断启动流程。"""

    class _BrokenStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("数据库被占用")

    monkeypatch.setattr(monitor, "ExperimentStore", _BrokenStore)
    monkeypatch.setattr(monitor, "_init_network", lambda cfg, args: (_FakeNetwork(), 1, 2))
    monkeypatch.setattr(
        monitor,
        "_init_logging",
        lambda cfg, args, network: (_FakeLogger(), object(), None, 0, None),
    )
    monkeypatch.setattr(monitor, "_start_tracking", lambda *args, **kwargs: (None, None))

    cfg, _args, _network, _logger, _observer, record, start_tick, end_tick, db_path = (
        monitor._prepare_run({"ticks": 2}, str(tmp_path / "x.db"))
    )

    assert cfg.ticks == 2
    assert record is None
    assert (start_tick, end_tick) == (1, 2)
    assert db_path == str(tmp_path / "x.db")


# ── 仿真线程 ──────────────────────────────────────────────────


def _default_finalize(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    """正常收尾替身：返回摘要、最终指标与输出路径。"""
    return (
        {"alive_neurons": 1},
        {"tick": kwargs["run_end_tick"]},
        "outputs/_fake/run.jsonl",
    )


def _run_thread(
    monkeypatch: Any,
    *,
    control: Any,
    network: _FakeNetwork | None = None,
    logger: _FakeLogger | None = None,
    end_tick: int = 3,
    finalize: Callable[..., Any] | None = None,
) -> tuple[_FakeLogger, dict[str, Any]]:
    """以替身驱动一次仿真线程主体，返回日志替身与收尾参数。"""
    monkeypatch.setattr(monitor, "AutoLearner", lambda **kwargs: _FakeLearner())
    monkeypatch.setattr(monitor, "_finalize_run", finalize or _default_finalize)
    finished: dict[str, Any] = {}
    monkeypatch.setattr(
        monitor,
        "_finish_tracking_in_thread",
        lambda db_path, record, **kwargs: finished.update(kwargs),
    )

    logger = logger or _FakeLogger()
    monitor._simulation_state["control"] = control
    monitor._simulation_state["message_queue"] = deque()

    monitor._run_simulation_thread(
        cfg=_FakeCfg(),  # type: ignore[arg-type]
        args=None,  # type: ignore[arg-type]
        network=network or _FakeNetwork(),  # type: ignore[arg-type]
        logger=logger,  # type: ignore[arg-type]
        observer=object(),  # type: ignore[arg-type]
        record=None,
        start_tick=1,
        end_tick=end_tick,
        db_path="unused.db",
    )
    return logger, finished


def test_run_thread_completes_and_releases_lock(monkeypatch: Any) -> None:
    """正常跑完应逐 tick 记录、标记 finished 并释放运行锁。"""
    logger, finished = _run_thread(monkeypatch, control=_FakeControl(speed_ms=1))

    assert logger.logged == [1, 2, 3]
    assert finished["status"] == "completed"
    assert finished["error"] is None
    assert finished["summary"] == {"alive_neurons": 1}
    assert finished["final_metrics"] == {"tick": 3}
    assert monitor._simulation_state["state"] == "finished"
    assert monitor._simulation_state["running"] is False


def test_run_thread_breaks_early_when_stopped(monkeypatch: Any) -> None:
    """收到停止请求时不执行任何 tick，直接进入收尾。"""
    logger, finished = _run_thread(monkeypatch, control=_FakeControl(stopped=True), end_tick=5)

    assert logger.logged == []
    assert finished["status"] == "completed"
    assert monitor._simulation_state["state"] == "finished"


def test_run_thread_resumes_after_pause(monkeypatch: Any) -> None:
    """暂停只阻塞一次轮询，恢复后继续执行全部 tick。"""
    logger, _finished = _run_thread(monkeypatch, control=_FakeControl(paused_once=True))

    assert logger.logged == [1, 2, 3]


def test_run_thread_marks_failed_when_tick_raises(monkeypatch: Any) -> None:
    """仿真中途异常应记录 failed、关闭日志并保留错误信息。"""
    logger, finished = _run_thread(
        monkeypatch,
        control=_FakeControl(),
        network=_FakeNetwork(fail_at=2),
    )

    assert logger.logged == [1]
    assert finished["status"] == "failed"
    assert "模拟 tick 失败" in (finished["error"] or "")
    assert logger.closed is True


def test_run_thread_marks_failed_when_finalize_raises(monkeypatch: Any) -> None:
    """收尾阶段异常应降级为 failed，并确保日志被关闭。"""

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("收尾失败")

    logger, finished = _run_thread(monkeypatch, control=_FakeControl(), finalize=_boom)

    assert finished["status"] == "failed"
    assert "收尾失败" in (finished["error"] or "")
    assert logger.closed is True
    assert monitor._simulation_state["running"] is False


def test_run_thread_survives_logger_close_failure(monkeypatch: Any) -> None:
    """日志关闭自身抛异常时，仿真线程仍需完成收尾且不向外抛。"""

    class _BadCloseLogger(_FakeLogger):
        def close(self) -> None:
            raise RuntimeError("日志关闭失败")

    logger, finished = _run_thread(
        monkeypatch,
        control=_FakeControl(),
        network=_FakeNetwork(fail_at=1),
        logger=_BadCloseLogger(),
    )

    assert isinstance(logger, _BadCloseLogger)
    assert finished["status"] == "failed"
    assert monitor._simulation_state["running"] is False


# ── REST 端点分支 ─────────────────────────────────────────────


def test_start_with_non_object_body_falls_back_to_empty(
    client: TestClient, monkeypatch: Any
) -> None:
    """请求体是数组时应降级为空字典，而不是抛出 500。"""
    captured: dict[str, Any] = {}

    def _prepare(body: dict[str, Any], db_path: str) -> Any:
        captured["body"] = body
        raise ValueError("拦截后续流程")

    monkeypatch.setattr(monitor, "_prepare_run", _prepare)

    resp = client.post("/api/run", json=[1, 2, 3])

    assert resp.status_code == 400
    assert captured["body"] == {}


def test_start_with_invalid_json_falls_back_to_empty(client: TestClient, monkeypatch: Any) -> None:
    """非法 JSON 请求体应降级为空字典。"""
    captured: dict[str, Any] = {}

    def _prepare(body: dict[str, Any], db_path: str) -> Any:
        captured["body"] = body
        raise ValueError("拦截后续流程")

    monkeypatch.setattr(monitor, "_prepare_run", _prepare)

    resp = client.post(
        "/api/run",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 400
    assert captured["body"] == {}


def test_start_returns_500_on_unexpected_prepare_error(
    client: TestClient, monkeypatch: Any
) -> None:
    """准备阶段出现非 ValueError 异常时返回 500 并保留原因。"""

    def _prepare(body: dict[str, Any], db_path: str) -> Any:
        raise RuntimeError("内部爆炸")

    monkeypatch.setattr(monitor, "_prepare_run", _prepare)

    resp = client.post("/api/run", json={})

    assert resp.status_code == 500
    assert "启动失败" in resp.json()["error"]
    assert monitor._simulation_state["running"] is False


def test_status_reports_paused(client: TestClient) -> None:
    """运行中被暂停时，状态查询应报告 paused。"""
    control = LiveControlState(tick_ms=0)
    control.pause()
    monitor._simulation_state.update(
        {"running": True, "control": control, "state": "running", "tick": 5}
    )

    data = client.get("/api/run/status").json()

    assert data["running"] is True
    assert data["state"] == "paused"
    assert data["tick"] == 5


def test_command_rejected_after_finished(client: TestClient) -> None:
    """仿真已结束时不再接受控制命令。"""
    monitor._simulation_state.update(
        {"running": True, "state": "finished", "control": LiveControlState()}
    )

    resp = client.post("/api/run/command", json={"type": "pause"})

    assert resp.status_code == 404
    assert "已结束" in resp.json()["error"]


def test_command_applies_speed_change(client: TestClient) -> None:
    """调速命令应写入控制状态并返回命令名。"""
    control = LiveControlState(tick_ms=50)
    monitor._simulation_state.update({"running": True, "state": "running", "control": control})

    resp = client.post("/api/run/command", json={"type": "speed_change", "tick_ms": 20})

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "command": "speed_change"}
    assert control.speed_ms == 20


def test_command_rejects_unknown_type(client: TestClient) -> None:
    """未知命令类型返回 400，不得影响控制状态。"""
    control = LiveControlState()
    monitor._simulation_state.update({"running": True, "state": "running", "control": control})

    resp = client.post("/api/run/command", json={"type": "bogus"})

    assert resp.status_code == 400
    assert "未知命令" in resp.json()["error"]
    assert control.paused is False


# ── WebSocket 端点 ────────────────────────────────────────────


def test_ws_sends_paused_snapshot_then_cleans_up(client: TestClient) -> None:
    """连接时应推送状态快照（暂停态归一为 paused），结束后清理全局状态。"""
    control = LiveControlState(tick_ms=0)
    control.pause()
    monitor._simulation_state.update(
        {
            "running": True,
            "control": control,
            "state": "running",
            "tick": 7,
            "thread": _finished_thread(),
            "message_queue": deque(),
        }
    )

    with client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "status"
        assert snapshot["state"] == "paused"
        assert snapshot["tick"] == 7
        # 标记结束，让服务端主循环在下次状态检查时退出
        monitor._simulation_state["state"] = "finished"

    assert monitor._simulation_state["running"] is False
    assert monitor._simulation_state["state"] == "idle"


def test_ws_flushes_queue_and_applies_commands(client: TestClient) -> None:
    """WebSocket 应排空消息队列，并响应合法命令、忽略非法命令。"""
    control = LiveControlState(tick_ms=50)
    queue: deque = deque()
    queue.append({"type": "tick_update", "tick": 1, "neurons": [], "connections": [], "stats": {}})
    monitor._simulation_state.update(
        {
            "running": True,
            "control": control,
            "state": "running",
            "tick": 1,
            "thread": _finished_thread(),
            "message_queue": queue,
        }
    )

    with client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "status"

        tick_msg = ws.receive_json()
        assert tick_msg["type"] == "tick_update"
        assert tick_msg["tick"] == 1

        ws.send_json({"type": "pause"})
        _wait_until(lambda: control.paused)

        ws.send_json({"type": "speed_change", "tick_ms": 10})
        _wait_until(lambda: control.speed_ms == 10)

        # 非法命令被静默丢弃，不影响后续合法命令
        ws.send_json({"type": "bogus"})
        ws.send_json({"type": "resume"})
        _wait_until(lambda: not control.paused)

        monitor._simulation_state["state"] = "finished"

    assert monitor._simulation_state["running"] is False


def test_ws_exits_immediately_when_already_finished(client: TestClient) -> None:
    """连接时仿真已结束且线程已退出，应发完积压消息后立即收尾。"""
    queue: deque = deque()
    queue.append({"type": "status", "state": "finished", "tick": 9, "tick_ms": 0})
    monitor._simulation_state.update(
        {
            "running": True,
            "control": None,
            "state": "finished",
            "thread": _finished_thread(),
            "message_queue": queue,
        }
    )

    with client.websocket_connect("/ws") as ws:
        message = ws.receive_json()
        assert message["type"] == "status"
        assert message["tick"] == 9

    assert monitor._simulation_state["running"] is False
    assert monitor._simulation_state["state"] == "idle"


def test_ws_polls_state_after_idle_timeout(client: TestClient) -> None:
    """无消息时主循环应超时轮询状态，并在结束后退出（不得挂死）。"""
    monitor._simulation_state.update(
        {
            "running": True,
            "control": None,
            "state": "running",
            "thread": _finished_thread(),
            "message_queue": deque(),
        }
    )

    with client.websocket_connect("/ws"):
        # 保持连接但不下发消息，强制主循环经历一次 1 秒超时轮询
        time.sleep(1.2)
        monitor._simulation_state["state"] = "finished"

    assert monitor._simulation_state["running"] is False


def test_ws_survives_iteration_error(client: TestClient) -> None:
    """主循环内出现异常时应被吞掉并清理客户端集合。"""

    class _BadQueue:
        """模拟损坏的消息队列：非空但取出即抛。"""

        def __bool__(self) -> bool:
            return True

        def popleft(self) -> Any:
            raise RuntimeError("队列损坏")

    monitor._simulation_state.update(
        {
            "running": True,
            "control": None,
            "state": "finished",
            "thread": _finished_thread(),
            "message_queue": _BadQueue(),
        }
    )

    with client.websocket_connect("/ws"):
        pass

    assert monitor._ws_clients == set()
    assert monitor._simulation_state["running"] is False
