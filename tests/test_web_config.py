"""Phase 3 配置管理 API 测试

验证配置 schema、校验、保存与预设管理端点。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurogenesis_lab.web.app import create_web_app
from neurogenesis_lab.web.routes import config as config_mod


@pytest.fixture
def client(tmp_path):
    """创建测试客户端。"""
    db = tmp_path / "test.db"
    app = create_web_app(db_path=str(db))
    return TestClient(app)


@pytest.fixture
def preset_dir(tmp_path, monkeypatch):
    """将预设保存目录重定向到临时目录，避免污染项目 experiments/。"""
    target = tmp_path / "presets"
    target.mkdir()
    monkeypatch.setattr(config_mod, "_EXPERIMENTS_DIR", target)
    return target


def test_get_schema(client):
    """配置 schema API 返回分组信息。"""
    data = client.get("/api/config/schema").json()
    assert "simulation" in data
    assert "neuron" in data
    assert "network" in data
    assert "evolution" in data
    assert "ticks" in data["simulation"]


def test_validate_config_valid(client):
    """合法配置校验通过。"""
    resp = client.post(
        "/api/config/validate",
        json={"simulation": {"ticks": 100, "seed": 42, "dt": 1.0}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["errors"] == []


def test_validate_config_invalid(client):
    """非法配置校验失败。"""
    resp = client.post(
        "/api/config/validate",
        json={"simulation": {"ticks": -1}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert len(body["errors"]) > 0


def test_validate_config_string_values(client):
    """字符串值（真实表单提交形态）校验通过且不抛 500。"""
    resp = client.post(
        "/api/config/validate",
        json={
            "simulation": {"ticks": "100", "seed": "42", "dt": "1.0"},
            "neuron": {"tau_m": "20.0"},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True, body
    assert body["errors"] == []


def test_validate_config_bad_json(client):
    """非 JSON 请求体返回错误。"""
    resp = client.post(
        "/api/config/validate",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    body = resp.json()
    assert body["valid"] is False


def test_save_config(client, preset_dir):
    """保存配置为 TOML 文件。"""
    resp = client.post(
        "/api/config/save",
        json={
            "name": "test_preset",
            "config": {"simulation": {"ticks": 100}},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "test_preset.toml" in body["path"]
    assert (preset_dir / "test_preset.toml").is_file()


def test_save_config_missing_name(client):
    """缺少 name 字段返回 400。"""
    resp = client.post("/api/config/save", json={"config": {}})
    assert resp.status_code == 400


def test_save_config_invalid_params(client):
    """非法参数返回 400。"""
    resp = client.post(
        "/api/config/save",
        json={"name": "bad", "config": {"simulation": {"ticks": -5}}},
    )
    assert resp.status_code == 400


def test_list_presets(client):
    """预设列表返回数组。"""
    data = client.get("/api/config/presets").json()
    assert isinstance(data, list)


def test_load_preset_existing(client, preset_dir):
    """加载已有预设文件。"""
    # 先保存一个
    client.post(
        "/api/config/save",
        json={"name": "loadable", "config": {"simulation": {"ticks": 80}}},
    )
    resp = client.get("/api/config/presets/loadable")
    assert resp.status_code == 200
    body = resp.json()
    # TOML 解析后顶层包含各分组键
    assert "ticks" in body or "simulation" in body


def test_load_preset_missing(client):
    """加载不存在的预设返回 404。"""
    resp = client.get("/api/config/presets/nonexistent_xyz")
    assert resp.status_code == 404
