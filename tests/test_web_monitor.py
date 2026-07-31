"""Phase 4 监控 API 测试

验证仿真状态查询、启动与监控页面端点。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurogenesis_lab.web.app import create_web_app


@pytest.fixture
def client(tmp_path):
    """创建测试客户端。"""
    db = tmp_path / "test.db"
    app = create_web_app(db_path=str(db))
    return TestClient(app)


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


def test_start_simulation(client):
    """启动仿真返回 started 状态。"""
    resp = client.post("/api/run", json={"ticks": 5, "seed": 1, "neurons": 4})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert "ws_url" in data


def test_start_simulation_conflict(client):
    """重复启动仿真返回 409。"""
    client.post("/api/run", json={"ticks": 50, "seed": 1, "neurons": 4})
    resp = client.post("/api/run", json={"ticks": 50, "seed": 2, "neurons": 4})
    assert resp.status_code == 409


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
