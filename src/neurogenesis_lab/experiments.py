"""实验管理 — SQLite 元数据存储、过滤查询与 JSON 索引导出 (Phase 8 / P2)

元数据字段与 ``outputs/experiments/index.json`` 的结构对齐：
顶层标量（experiment_id、seed、ticks 等）映射为 SQLite 列以支持索引，
嵌套结构（config、cli_args、summary、final_metrics）以 JSON 文本列存储。
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

from neurogenesis_lab.config import SCHEMA_VERSION

# 过滤操作符 — 双字符操作符须先于单字符匹配
_FILTER_OPS: tuple[str, ...] = ("<=", ">=", "!=", "<", ">", "=")

# 默认数据库路径（相对工作目录）
DEFAULT_DB_PATH = Path("outputs") / "experiments.db"


def utc_now_iso() -> str:
    """返回 UTC 当前时间的 ISO 字符串（与 index.json 格式一致）。"""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ExperimentRecord:
    """单次实验的元数据记录 — 字段与 index.json 条目对齐。

    Attributes:
        experiment_id: 唯一实验 ID（输出目录名，如 run_20260306_152157_022）。
        experiment_name: 用户指定的实验名称，可为 None。
        command: 产生记录的 CLI 命令，默认 "run"。
        status: 实验状态（running / completed / failed）。
        schema_version: 元数据 Schema 版本。
        version: 产生记录的软件版本号。
        created_at: 记录创建时间（UTC ISO）。
        updated_at: 记录最后更新时间（UTC ISO）。
        started_at: 仿真开始时间（UTC ISO）。
        finished_at: 仿真结束时间（UTC ISO），未结束为 None。
        seed: 随机种子。
        ticks: 仿真步数。
        neurons: 初始神经元数。
        runtime_seconds: 仿真耗时（秒），未结束为 None。
        output_dir: 输出目录路径。
        output_file: 主输出文件路径（run.jsonl）。
        config: 完整 SimConfig 字典。
        cli_args: 命令行参数字典。
        summary: 引擎摘要（get_summary() 结果）。
        final_metrics: 结束时的网络级聚合指标。
        error: 失败时的错误信息，正常为 None。
    """

    experiment_id: str
    experiment_name: str | None = None
    command: str = "run"
    status: str = "running"
    schema_version: str = SCHEMA_VERSION
    version: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    started_at: str = field(default_factory=utc_now_iso)
    finished_at: str | None = None
    seed: int | None = None
    ticks: int | None = None
    neurons: int | None = None
    runtime_seconds: float | None = None
    output_dir: str | None = None
    output_file: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    cli_args: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典（用于 JSON 索引导出）。"""
        return asdict(self)

    def flatten(self) -> dict[str, Any]:
        """展平为单层字典 — 顶层标量 + summary/final_metrics 指标，用于过滤匹配。"""
        flat: dict[str, Any] = {
            key: value for key, value in asdict(self).items() if not isinstance(value, dict)
        }
        # 指标键直接暴露（final_metrics 覆盖 summary 中的同名键，取结束时刻值）
        flat.update(self.summary)
        flat.update(self.final_metrics)
        return flat


def parse_filter_expression(expression: str) -> list[tuple[str, str, Any]]:
    """解析过滤表达式为 (key, op, value) 三元组列表。

    语法：``key op value``，op 支持 >, <, >=, <=, =, !=；
    多条件用 ``,`` 或 `` and `` 连接（AND 语义）。

    Args:
        expression: 过滤表达式，如 "neurons>100,seed=42"。

    Returns:
        条件三元组列表。

    Raises:
        ValueError: 表达式语法非法时。
    """
    conditions: list[tuple[str, str, Any]] = []
    normalized = expression.replace(" and ", ",")
    for clause in normalized.split(","):
        clause = clause.strip()
        if not clause:
            continue
        for op in _FILTER_OPS:
            idx = clause.find(op)
            if idx > 0:
                key = clause[:idx].strip()
                raw_value = clause[idx + len(op) :].strip()
                if not key or not raw_value:
                    raise ValueError(f"过滤条件缺少键或值: {clause!r}")
                conditions.append((key, op, _coerce_value(raw_value)))
                break
        else:
            raise ValueError(f"过滤条件缺少操作符 (>, <, >=, <=, =, !=): {clause!r}")
    if not conditions:
        raise ValueError(f"过滤表达式为空: {expression!r}")
    return conditions


def _coerce_value(raw: str) -> Any:
    """将过滤值字符串转换为 int / float / bool / str。"""
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw.strip("'\"")


def _match_condition(actual: Any, op: str, expected: Any) -> bool:
    """判断单个条件是否成立 — 键缺失或类型不可比时视为不匹配。"""
    if actual is None:
        return False
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return False
        if op == ">":
            return bool(actual > expected)
        if op == "<":
            return bool(actual < expected)
        if op == ">=":
            return bool(actual >= expected)
        if op == "<=":
            return bool(actual <= expected)
    if op == "=":
        return bool(actual == expected)
    if op == "!=":
        return bool(actual != expected)
    return False


def matches_filters(record: ExperimentRecord, conditions: list[tuple[str, str, Any]]) -> bool:
    """判断实验记录是否满足全部过滤条件（AND 语义）。"""
    flat = record.flatten()
    return all(_match_condition(flat.get(key), op, expected) for key, op, expected in conditions)


# ── SQLite 存储 ──────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id       TEXT PRIMARY KEY,
    experiment_name     TEXT,
    command             TEXT NOT NULL DEFAULT 'run',
    status              TEXT NOT NULL DEFAULT 'running',
    schema_version      TEXT NOT NULL,
    version             TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    seed                INTEGER,
    ticks               INTEGER,
    neurons             INTEGER,
    runtime_seconds     REAL,
    output_dir          TEXT,
    output_file         TEXT,
    config_json         TEXT NOT NULL DEFAULT '{}',
    cli_args_json       TEXT NOT NULL DEFAULT '{}',
    summary_json        TEXT NOT NULL DEFAULT '{}',
    final_metrics_json  TEXT NOT NULL DEFAULT '{}',
    error               TEXT
)
"""

_CREATE_INDEX_SQLS = (
    "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments (status)",
    "CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments (experiment_name)",
)

# 旧原型 schema（id/name/elapsed 列）→ 新 schema 的列映射
_LEGACY_COLUMNS = {"id", "name", "elapsed"}


class ExperimentStore:
    """实验元数据的 SQLite 存储 — 支持增删查、过滤与 JSON 索引导出。"""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        """打开（必要时创建/迁移）实验数据库。

        Args:
            db_path: SQLite 数据库文件路径。
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    def _migrate(self) -> None:
        """建表并在检测到旧原型 schema 时迁移历史数据。"""
        columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(experiments)").fetchall()
        }
        if columns and _LEGACY_COLUMNS <= columns:
            self._conn.execute("ALTER TABLE experiments RENAME TO experiments_legacy")
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO experiments (
                    experiment_id, experiment_name, command, status, schema_version,
                    version, created_at, updated_at, started_at, seed, ticks, neurons,
                    runtime_seconds, output_dir, config_json, summary_json
                )
                SELECT
                    id, name, 'run', COALESCE(status, 'completed'), ?,
                    COALESCE(version, ''), created_at, created_at, created_at,
                    seed, ticks, neurons, elapsed, output_dir,
                    COALESCE(config_json, '{}'), COALESCE(summary_json, '{}')
                FROM experiments_legacy
                """,
                (SCHEMA_VERSION,),
            )
            self._conn.execute("DROP TABLE experiments_legacy")
        else:
            self._conn.execute(_CREATE_TABLE_SQL)
        for sql in _CREATE_INDEX_SQLS:
            self._conn.execute(sql)
        self._conn.commit()

    def save(self, record: ExperimentRecord) -> None:
        """插入或覆盖一条实验记录（按 experiment_id upsert）。"""
        record.updated_at = utc_now_iso()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                experiment_id, experiment_name, command, status, schema_version,
                version, created_at, updated_at, started_at, finished_at,
                seed, ticks, neurons, runtime_seconds, output_dir, output_file,
                config_json, cli_args_json, summary_json, final_metrics_json, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.experiment_id,
                record.experiment_name,
                record.command,
                record.status,
                record.schema_version,
                record.version,
                record.created_at,
                record.updated_at,
                record.started_at,
                record.finished_at,
                record.seed,
                record.ticks,
                record.neurons,
                record.runtime_seconds,
                record.output_dir,
                record.output_file,
                json.dumps(record.config, ensure_ascii=False),
                json.dumps(record.cli_args, ensure_ascii=False),
                json.dumps(record.summary, ensure_ascii=False),
                json.dumps(record.final_metrics, ensure_ascii=False),
                record.error,
            ),
        )
        self._conn.commit()

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        """按 ID 查询实验记录，不存在时返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_experiments(
        self, filter_expression: str | None = None, limit: int = 0
    ) -> list[ExperimentRecord]:
        """列出实验记录（按创建时间倒序），可选过滤表达式与条数上限。

        Args:
            filter_expression: 过滤表达式（见 parse_filter_expression），None 表示不过滤。
            limit: 最多返回条数，0 表示不限。

        Returns:
            满足条件的实验记录列表。
        """
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC, experiment_id DESC"
        ).fetchall()
        records = [self._row_to_record(row) for row in rows]
        if filter_expression:
            conditions = parse_filter_expression(filter_expression)
            records = [r for r in records if matches_filters(r, conditions)]
        if limit > 0:
            records = records[:limit]
        return records

    def delete(self, experiment_id: str) -> bool:
        """删除一条实验记录，返回是否实际删除。"""
        cursor = self._conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def export_index(self, path: Path | str) -> Path:
        """将全部记录导出为 index.json 格式（JSON 导出/迁移格式）。

        Args:
            path: 导出文件路径。

        Returns:
            实际写入的 Path 对象。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "experiments": [record.to_dict() for record in self.list_experiments()],
        }
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return target

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ExperimentRecord:
        """将 SQLite 行转换为 ExperimentRecord。"""
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            experiment_name=row["experiment_name"],
            command=row["command"],
            status=row["status"],
            schema_version=row["schema_version"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            seed=row["seed"],
            ticks=row["ticks"],
            neurons=row["neurons"],
            runtime_seconds=row["runtime_seconds"],
            output_dir=row["output_dir"],
            output_file=row["output_file"],
            config=json.loads(row["config_json"] or "{}"),
            cli_args=json.loads(row["cli_args_json"] or "{}"),
            summary=json.loads(row["summary_json"] or "{}"),
            final_metrics=json.loads(row["final_metrics_json"] or "{}"),
            error=row["error"],
        )

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


# ── 公共工具函数（供 CLI 和 Web API 共享使用）──────────────────


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """将嵌套配置字典展平为点分键（如 network.max_neurons）。"""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = value
    return flat


def config_diff_keys(configs: list[dict[str, Any]]) -> list[str]:
    """收集在各实验间取值不同的配置键。"""
    all_keys = sorted({key for cfg in configs for key in cfg})
    diff_keys = []
    for key in all_keys:
        values = [cfg.get(key) for cfg in configs]
        if any(v != values[0] for v in values[1:]):
            diff_keys.append(key)
    return diff_keys


def remove_output_dir(experiment_id: str, output_dir: str | None) -> bool:
    """安全删除实验输出目录 — 仅当目录名与实验 ID 一致时执行。"""
    import shutil

    if not output_dir:
        return False
    path = Path(output_dir)
    if not path.is_dir():
        return False
    if path.name != experiment_id:
        print(f"  跳过输出目录（目录名与实验 ID 不一致，拒绝删除）: {path}")
        return False
    shutil.rmtree(path)
    return True
