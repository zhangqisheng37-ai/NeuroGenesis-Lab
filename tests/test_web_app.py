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


def test_all_pages_include_version(client):
    """所有页面导航/页脚均显示版本号（全局模板变量注入）。"""
    from neurogenesis_lab import __version__

    for path in ("/", "/experiments", "/configure", "/monitor", "/benchmark"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} 返回 {resp.status_code}"
        assert f"v{__version__}" in resp.text, f"{path} 缺少版本号"


def test_favicon_no_404(client):
    """favicon 请求不返回 404。"""
    resp = client.get("/favicon.ico")
    assert resp.status_code == 204


def test_dashboard_run_card_points_to_workbench(client):
    """仪表盘「运行实验」卡片指向实验工作台。"""
    resp = client.get("/")
    # 快速操作区中 href=/monitor 的卡片存在
    assert '/monitor"' in resp.text
