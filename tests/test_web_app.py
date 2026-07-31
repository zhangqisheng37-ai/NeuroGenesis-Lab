"""P5 Web 管理界面基础测试"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurogenesis_lab.web.app import create_web_app


@pytest.fixture
def client(tmp_path):
    """创建使用临时数据库的测试客户端。"""
    db = tmp_path / "test.db"
    app = create_web_app(db_path=str(db))
    return TestClient(app)


def test_create_app():
    """应用工厂可正常创建 FastAPI 实例。"""
    app = create_web_app()
    assert app is not None
    assert app.title == "NeuroGenesis Lab Web"


def test_dashboard_returns_200(client):
    """仪表盘首页返回 200。"""
    response = client.get("/")
    assert response.status_code == 200
    assert "NeuroGenesis Lab" in response.text


def test_static_css_accessible(client):
    """全局 CSS 静态文件可访问。"""
    response = client.get("/static/css/app.css")
    assert response.status_code == 200
    assert "--accent" in response.text


def test_static_js_accessible(client):
    """全局 JS 静态文件可访问。"""
    response = client.get("/static/js/app.js")
    assert response.status_code == 200
    assert "apiGet" in response.text
