"""Phase 2 实验管理 API 测试

验证实验列表、详情、删除与对比端点的正确性。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore
from neurogenesis_lab.web.app import create_web_app


@pytest.fixture
def client_with_data(tmp_path):
    """创建带有测试数据的 TestClient。"""
    db = tmp_path / "test.db"
    app = create_web_app(db_path=str(db))

    # 插入两条测试记录
    with ExperimentStore(str(db)) as store:
        store.save(
            ExperimentRecord(
                experiment_id="test_exp_001",
                experiment_name="测试实验1",
                status="completed",
                seed=42,
                ticks=100,
                neurons=8,
                runtime_seconds=1.5,
                config={"simulation": {"ticks": 100}, "neuron": {"threshold": 0.5}},
                summary={"avg_fitness": 0.75},
                final_metrics={"survival_rate": 0.9},
            )
        )
        store.save(
            ExperimentRecord(
                experiment_id="test_exp_002",
                experiment_name="测试实验2",
                status="completed",
                seed=123,
                ticks=200,
                neurons=16,
                runtime_seconds=3.0,
                config={"simulation": {"ticks": 200}, "neuron": {"threshold": 0.8}},
                summary={"avg_fitness": 0.85},
                final_metrics={"survival_rate": 0.95},
            )
        )

    return TestClient(app)


def test_experiments_page_returns_200(client_with_data):
    """实验列表页面返回 200。"""
    response = client_with_data.get("/experiments")
    assert response.status_code == 200
    assert "实验列表" in response.text


def test_api_experiments_list(client_with_data):
    """实验列表 API 返回分页格式的记录。"""
    data = client_with_data.get("/api/experiments").json()
    assert "experiments" in data
    assert "total" in data
    assert "offset" in data
    assert "limit" in data
    experiments = data["experiments"]
    assert len(experiments) == 2
    assert data["total"] == 2
    ids = {r["experiment_id"] for r in experiments}
    assert "test_exp_001" in ids
    assert "test_exp_002" in ids


def test_api_experiment_detail(client_with_data):
    """实验详情 API 返回正确记录。"""
    data = client_with_data.get("/api/experiments/test_exp_001").json()
    assert data["experiment_id"] == "test_exp_001"
    assert data["seed"] == 42
    assert data["status"] == "completed"


def test_api_experiment_not_found(client_with_data):
    """查询不存在的实验返回 404。"""
    response = client_with_data.get("/api/experiments/nonexistent")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_api_delete_experiment(client_with_data):
    """删除实验 API 移除记录并返回确认。"""
    response = client_with_data.delete("/api/experiments/test_exp_001")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"
    assert body["id"] == "test_exp_001"

    # 确认已被删除
    get_resp = client_with_data.get("/api/experiments/test_exp_001")
    assert get_resp.status_code == 404


def test_api_delete_nonexistent(client_with_data):
    """删除不存在的实验返回 404。"""
    response = client_with_data.delete("/api/experiments/nonexistent")
    assert response.status_code == 404


def test_api_compare_experiments(client_with_data):
    """实验对比 API 返回差异键与展平配置。"""
    response = client_with_data.get(
        "/api/experiments/compare",
        params={"ids": "test_exp_001,test_exp_002"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "records" in body
    assert "diff_keys" in body
    assert "flat_configs" in body
    assert len(body["records"]) == 2
    # simulation.ticks 和 neuron.threshold 应不同
    assert "simulation.ticks" in body["diff_keys"]
    assert "neuron.threshold" in body["diff_keys"]


def test_experiments_filter(client_with_data):
    """过滤参数可正确筛选实验列表。"""
    response = client_with_data.get(
        "/api/experiments",
        params={"filter": "seed=42"},
    )
    assert response.status_code == 200
    data = response.json()
    experiments = data["experiments"]
    assert len(experiments) == 1
    assert experiments[0]["experiment_id"] == "test_exp_001"


def test_experiments_filter_no_match(client_with_data):
    """过滤无匹配结果时返回空列表。"""
    response = client_with_data.get(
        "/api/experiments",
        params={"filter": "neurons>100"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["experiments"]) == 0


def test_experiments_limit(client_with_data):
    """limit 参数可限制返回条数。"""
    response = client_with_data.get("/api/experiments", params={"limit": 1})
    assert response.status_code == 200
    data = response.json()
    assert len(data["experiments"]) == 1
    assert data["total"] == 2
    assert data["limit"] == 1
    assert data["offset"] == 0


def test_experiments_offset(client_with_data):
    """offset 参数可跳过记录。"""
    # 先获取总数
    all_data = client_with_data.get("/api/experiments").json()
    total = all_data["total"]
    # 跳过第一条
    page2 = client_with_data.get("/api/experiments", params={"offset": 1, "limit": 1}).json()
    assert len(page2["experiments"]) == 1
    assert page2["offset"] == 1
    assert page2["total"] == total
    # 确认偏移后的记录与第一页不同
    assert page2["experiments"][0]["experiment_id"] != all_data["experiments"][0]["experiment_id"]


def test_experiments_sort_by(client_with_data):
    """sort_by 和 sort_order 参数可正确排序。"""
    # 按 seed 升序
    response = client_with_data.get(
        "/api/experiments",
        params={"sort_by": "seed", "sort_order": "asc"},
    )
    assert response.status_code == 200
    data = response.json()
    seeds = [e["seed"] for e in data["experiments"]]
    assert seeds == sorted(seeds)

    # 按 seed 降序
    response = client_with_data.get(
        "/api/experiments",
        params={"sort_by": "seed", "sort_order": "desc"},
    )
    data = response.json()
    seeds = [e["seed"] for e in data["experiments"]]
    assert seeds == sorted(seeds, reverse=True)


def test_experiments_sort_invalid_field(client_with_data):
    """无效的 sort_by 回退到默认 created_at。"""
    response = client_with_data.get(
        "/api/experiments",
        params={"sort_by": "invalid_field"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["experiments"]) == 2


def test_experiment_detail_page(client_with_data):
    """实验详情页面返回 200 并包含实验 ID。"""
    response = client_with_data.get("/experiments/test_exp_001")
    assert response.status_code == 200
    assert "test_exp_001" in response.text


def test_experiment_detail_page_not_found(client_with_data):
    """不存在的实验详情页面返回 404。"""
    response = client_with_data.get("/experiments/nonexistent")
    assert response.status_code == 404


def test_experiments_compare_page(client_with_data):
    """实验对比页面返回 200 并包含对比内容。"""
    response = client_with_data.get(
        "/experiments/compare",
        params={"ids": "test_exp_001,test_exp_002"},
    )
    assert response.status_code == 200
    assert "实验对比" in response.text


def test_api_experiment_output_not_found(tmp_path):
    """不存在的实验输出 API 返回 404。"""
    db = tmp_path / "test.db"
    app = create_web_app(db_path=str(db))
    client = TestClient(app)
    response = client.get("/api/experiments/nonexistent/output")
    assert response.status_code == 404


def test_api_experiment_output_no_file(client_with_data):
    """无输出文件时返回空结果。"""
    response = client_with_data.get("/api/experiments/test_exp_001/output")
    assert response.status_code == 200
    data = response.json()
    assert "lines" in data
    assert "total_lines" in data
    assert "truncated" in data
    assert data["lines"] == []
    assert data["total_lines"] == 0


def test_api_experiment_output_with_file(tmp_path):
    """有输出文件时返回解析后的 JSONL 数据。"""
    import json

    db = tmp_path / "test.db"
    output_dir = tmp_path / "run_test"
    output_dir.mkdir()
    output_file = output_dir / "run.jsonl"
    # 写入测试 JSONL
    lines = [
        {"tick": 1, "alive": 8},
        {"tick": 2, "alive": 7},
        {"tick": 3, "alive": 6},
    ]
    with output_file.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore

    with ExperimentStore(str(db)) as store:
        store.save(
            ExperimentRecord(
                experiment_id="run_test",
                status="completed",
                output_file=str(output_file),
                config={},
            )
        )

    app = create_web_app(db_path=str(db))
    client = TestClient(app)
    response = client.get("/api/experiments/run_test/output")
    assert response.status_code == 200
    data = response.json()
    assert data["total_lines"] == 3
    assert data["truncated"] is False
    assert len(data["lines"]) == 3
    assert data["lines"][0]["tick"] == 1


def test_api_experiment_output_truncated(tmp_path):
    """limit 参数可截断输出。"""
    import json

    db = tmp_path / "test.db"
    output_dir = tmp_path / "run_trunc"
    output_dir.mkdir()
    output_file = output_dir / "run.jsonl"
    with output_file.open("w") as f:
        for i in range(10):
            f.write(json.dumps({"tick": i}) + "\n")

    from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore

    with ExperimentStore(str(db)) as store:
        store.save(
            ExperimentRecord(
                experiment_id="run_trunc",
                status="completed",
                output_file=str(output_file),
                config={},
            )
        )

    app = create_web_app(db_path=str(db))
    client = TestClient(app)
    response = client.get("/api/experiments/run_trunc/output", params={"limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert len(data["lines"]) == 3
    assert data["total_lines"] == 10
    assert data["truncated"] is True
