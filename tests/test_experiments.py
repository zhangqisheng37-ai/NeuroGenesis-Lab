"""单元测试 — 实验管理模块（SQLite 存储、过滤查询、索引导出）"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.experiments import (
    ExperimentRecord,
    ExperimentStore,
    matches_filters,
    parse_filter_expression,
)


def _make_record(exp_id: str = "run_20260731_000001_001", **overrides: object) -> ExperimentRecord:
    defaults: dict[str, object] = {
        "experiment_name": "demo",
        "status": "completed",
        "version": "0.6.1",
        "seed": 42,
        "ticks": 100,
        "neurons": 8,
        "runtime_seconds": 0.5,
        "output_dir": f"outputs/{exp_id}",
        "output_file": f"outputs/{exp_id}/run.jsonl",
        "config": {"ticks": 100, "network": {"max_neurons": 50}},
        "cli_args": {"command": "run", "neurons": 8},
        "summary": {"total_neurons": 33, "alive_neurons": 33, "total_spikes": 24},
        "final_metrics": {"tick": 100, "mean_energy": 0.85, "active_ratio": 1.0},
    }
    defaults.update(overrides)
    return ExperimentRecord(experiment_id=exp_id, **defaults)  # type: ignore[arg-type]


# ── 过滤表达式解析 ──────────────────────────────────────────


def test_parse_filter_single_condition() -> None:
    assert parse_filter_expression("neurons>100") == [("neurons", ">", 100)]


def test_parse_filter_multiple_conditions_comma_and_and() -> None:
    assert parse_filter_expression("neurons>100,seed=42") == [
        ("neurons", ">", 100),
        ("seed", "=", 42),
    ]
    assert parse_filter_expression("ticks>=50 and status=completed") == [
        ("ticks", ">=", 50),
        ("status", "=", "completed"),
    ]


def test_parse_filter_all_operators() -> None:
    for op in (">", "<", ">=", "<=", "=", "!="):
        assert parse_filter_expression(f"ticks{op}10") == [("ticks", op, 10)]


def test_parse_filter_value_coercion() -> None:
    assert parse_filter_expression("mean_energy>0.5") == [("mean_energy", ">", 0.5)]
    assert parse_filter_expression("stopped_early=true") == [("stopped_early", "=", True)]
    assert parse_filter_expression("experiment_name=demo") == [("experiment_name", "=", "demo")]


def test_parse_filter_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_filter_expression("neurons")
    with pytest.raises(ValueError):
        parse_filter_expression("=42")
    with pytest.raises(ValueError):
        parse_filter_expression("neurons>")
    with pytest.raises(ValueError):
        parse_filter_expression("  ,  ")


def test_matches_filters_on_flattened_metrics() -> None:
    record = _make_record()
    assert matches_filters(record, parse_filter_expression("alive_neurons>30"))
    assert matches_filters(record, parse_filter_expression("mean_energy<1.0,seed=42"))
    assert not matches_filters(record, parse_filter_expression("alive_neurons>100"))
    # 键缺失视为不匹配
    assert not matches_filters(record, parse_filter_expression("nonexistent_key>1"))
    # 字符串仅支持 = / !=
    assert matches_filters(record, parse_filter_expression("status!=failed"))
    assert not matches_filters(record, parse_filter_expression("status>completed"))


# ── SQLite 存储 CRUD ────────────────────────────────────────


def test_store_save_and_get_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    record = _make_record()
    with ExperimentStore(db) as store:
        store.save(record)
        loaded = store.get(record.experiment_id)
    assert loaded is not None
    assert loaded.experiment_id == record.experiment_id
    assert loaded.experiment_name == "demo"
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.config == record.config
    assert loaded.summary == record.summary
    assert loaded.final_metrics == record.final_metrics
    assert loaded.cli_args == record.cli_args


def test_store_get_missing_returns_none(tmp_path: Path) -> None:
    with ExperimentStore(tmp_path / "experiments.db") as store:
        assert store.get("no_such_experiment") is None


def test_store_save_upserts(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    record = _make_record(status="running")
    with ExperimentStore(db) as store:
        store.save(record)
        record.status = "completed"
        record.runtime_seconds = 1.23
        store.save(record)
        assert len(store.list_experiments()) == 1
        loaded = store.get(record.experiment_id)
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.runtime_seconds == 1.23


def test_store_list_with_filter_and_limit(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    with ExperimentStore(db) as store:
        store.save(_make_record("run_a", seed=1, created_at="2026-07-31T00:00:01Z"))
        store.save(_make_record("run_b", seed=2, created_at="2026-07-31T00:00:02Z"))
        store.save(_make_record("run_c", seed=3, created_at="2026-07-31T00:00:03Z"))
        all_records = store.list_experiments()
        filtered = store.list_experiments("seed>=2")
        limited = store.list_experiments(limit=2)
    # 按创建时间倒序
    assert [r.experiment_id for r in all_records] == ["run_c", "run_b", "run_a"]
    assert {r.experiment_id for r in filtered} == {"run_b", "run_c"}
    assert len(limited) == 2


def test_store_delete(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    with ExperimentStore(db) as store:
        store.save(_make_record("run_x"))
        assert store.delete("run_x") is True
        assert store.delete("run_x") is False
        assert store.get("run_x") is None


def test_store_export_index_json(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    index_path = tmp_path / "experiments" / "index.json"
    with ExperimentStore(db) as store:
        store.save(_make_record("run_a"))
        store.save(_make_record("run_b"))
        store.export_index(index_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "updated_at" in payload
    assert len(payload["experiments"]) == 2
    entry = payload["experiments"][0]
    # 与 index.json 结构对齐的关键字段
    for key in (
        "experiment_id",
        "experiment_name",
        "status",
        "config",
        "summary",
        "final_metrics",
        "runtime_seconds",
        "output_dir",
    ):
        assert key in entry


# ── 旧原型 schema 迁移 ──────────────────────────────────────


def test_store_migrates_legacy_schema(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE experiments (
            id          TEXT PRIMARY KEY,
            name        TEXT,
            created_at  TEXT NOT NULL,
            seed        INTEGER,
            ticks       INTEGER,
            neurons     INTEGER,
            config_json TEXT,
            summary_json TEXT,
            elapsed     REAL,
            output_dir  TEXT,
            version     TEXT,
            status      TEXT DEFAULT 'running'
        )
        """
    )
    conn.execute(
        "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "run_legacy_001",
            "legacy-exp",
            "2026-03-06T00:00:00Z",
            7,
            50,
            4,
            '{"ticks": 50}',
            '{"alive_neurons": 4}',
            0.42,
            "outputs/run_legacy_001",
            "0.6.0",
            "completed",
        ),
    )
    conn.commit()
    conn.close()

    with ExperimentStore(db) as store:
        migrated = store.get("run_legacy_001")
        assert migrated is not None
        assert migrated.experiment_name == "legacy-exp"
        assert migrated.status == "completed"
        assert migrated.seed == 7
        assert migrated.runtime_seconds == 0.42
        assert migrated.config == {"ticks": 50}
        assert migrated.summary == {"alive_neurons": 4}
        # 迁移后可正常写入新记录
        store.save(_make_record("run_new_001"))
        assert len(store.list_experiments()) == 2

    # 旧表已清理
    conn = sqlite3.connect(db)
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master").fetchall()}
    conn.close()
    assert "experiments_legacy" not in names


def test_record_flatten_merges_metrics() -> None:
    record = _make_record()
    flat = record.flatten()
    assert flat["seed"] == 42
    assert flat["total_spikes"] == 24
    assert flat["mean_energy"] == 0.85  # final_metrics 覆盖 summary
    assert "config" not in flat
