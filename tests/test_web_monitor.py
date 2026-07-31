"""Phase 4 监控 API 测试

验证仿真状态查询、启动与监控页面端点。
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from neurogenesis_lab.web.app import create_web_app


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
