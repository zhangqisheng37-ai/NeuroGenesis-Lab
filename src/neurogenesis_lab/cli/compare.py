"""CLI compare 命令 — 对比多个实验的参数差异与结果指标"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from neurogenesis_lab.experiments import (
    ExperimentRecord,
    ExperimentStore,
    config_diff_keys,
    flatten_config,
)


def _format_value(value: Any) -> str:
    """将指标值格式化为对齐友好的字符串。"""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_table(
    title: str, keys: list[str], columns: list[dict[str, Any]], headers: list[str]
) -> None:
    """打印键 × 实验的对比表格。"""
    print(f"\n{title}")
    rows = [[key, *(_format_value(col.get(key)) for col in columns)] for key in keys]
    all_headers = ["参数/指标", *headers]
    widths = [
        max(len(all_headers[i]), *(len(row[i]) for row in rows)) if rows else len(all_headers[i])
        for i in range(len(all_headers))
    ]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(all_headers)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def cmd_compare(args: argparse.Namespace) -> None:
    """对比多个实验：输出参数差异表与结果指标对比表。"""
    experiment_ids: list[str] = list(getattr(args, "experiment_ids", []) or [])
    if len(experiment_ids) < 2:
        print("compare 至少需要 2 个实验 ID")
        sys.exit(1)
    db_path = Path(getattr(args, "db", "outputs/experiments.db"))
    if not db_path.exists():
        print(f"实验数据库不存在: {db_path}")
        sys.exit(1)
    records: list[ExperimentRecord] = []
    with ExperimentStore(db_path) as store:
        for exp_id in experiment_ids:
            record = store.get(exp_id)
            if record is None:
                print(f"实验不存在: {exp_id}")
                sys.exit(1)
            records.append(record)

    headers = [r.experiment_name or r.experiment_id for r in records]
    print(f"实验对比 ({len(records)} 个实验)")
    for record in records:
        name = f" ({record.experiment_name})" if record.experiment_name else ""
        print(
            f"  - {record.experiment_id}{name}: status={record.status}, "
            f"seed={record.seed}, ticks={record.ticks}"
        )

    # 参数差异（仅展示取值不同的配置项）
    flat_configs = [flatten_config(r.config) for r in records]
    diff_keys = config_diff_keys(flat_configs)
    if diff_keys:
        _print_table("参数差异:", diff_keys, flat_configs, headers)
    else:
        print("\n参数差异: 无（各实验配置一致）")

    # 结果指标对比（summary + final_metrics 合并）
    metric_columns = [{**r.summary, **r.final_metrics} for r in records]
    metric_keys = sorted(
        {
            key
            for col in metric_columns
            for key, value in col.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    if metric_keys:
        _print_table("结果指标对比:", metric_keys, metric_columns, headers)
    else:
        print("\n结果指标对比: 无可用指标（实验可能尚未完成）")
