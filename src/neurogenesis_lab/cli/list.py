"""CLI list 命令 — 列出已追踪的实验"""

from __future__ import annotations

import argparse
from pathlib import Path

from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore


def _format_row(record: ExperimentRecord) -> tuple[str, ...]:
    """将实验记录格式化为表格行。"""
    alive = record.summary.get("alive_neurons")
    total = record.summary.get("total_neurons")
    survival = f"{alive}/{total}" if alive is not None and total is not None else "-"
    runtime = f"{record.runtime_seconds:.2f}" if record.runtime_seconds is not None else "-"
    return (
        record.experiment_id,
        record.experiment_name or "-",
        record.status,
        record.created_at,
        str(record.seed) if record.seed is not None else "-",
        str(record.ticks) if record.ticks is not None else "-",
        survival,
        runtime,
    )


def cmd_list(args: argparse.Namespace) -> None:
    """列出实验元数据，支持过滤表达式。"""
    db_path = Path(getattr(args, "db", "outputs/experiments.db"))
    if not db_path.exists():
        print(f"实验数据库不存在: {db_path}")
        print("提示: 先执行 run 命令产生实验记录")
        return
    filter_expr = getattr(args, "filter", None)
    limit = max(0, int(getattr(args, "limit", 0) or 0))
    try:
        with ExperimentStore(db_path) as store:
            records = store.list_experiments(filter_expr, limit=limit)
    except ValueError as exc:
        print(f"过滤表达式错误: {exc}")
        return
    if not records:
        print("无匹配实验" + (f" (filter: {filter_expr})" if filter_expr else ""))
        return

    headers = ("实验ID", "名称", "状态", "创建时间", "seed", "ticks", "存活/总数", "用时(s)")
    rows = [_format_row(r) for r in records]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(f"共 {len(records)} 个实验" + (f" (filter: {filter_expr})" if filter_expr else ""))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
