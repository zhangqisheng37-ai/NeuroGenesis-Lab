"""CLI delete 命令 — 删除实验元数据与输出文件"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from neurogenesis_lab.experiments import ExperimentStore


def _remove_output_dir(experiment_id: str, output_dir: str | None) -> bool:
    """安全删除实验输出目录 — 仅当目录名与实验 ID 一致时执行。"""
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


def cmd_delete(args: argparse.Namespace) -> None:
    """删除一个或多个实验的元数据（默认连同输出文件一起删除）。"""
    experiment_ids: list[str] = list(getattr(args, "experiment_ids", []) or [])
    db_path = Path(getattr(args, "db", "outputs/experiments.db"))
    keep_files = bool(getattr(args, "keep_files", False))
    if not db_path.exists():
        print(f"实验数据库不存在: {db_path}")
        sys.exit(1)
    deleted = 0
    with ExperimentStore(db_path) as store:
        for exp_id in experiment_ids:
            record = store.get(exp_id)
            if record is None:
                print(f"实验不存在，跳过: {exp_id}")
                continue
            files_removed = False
            if not keep_files:
                files_removed = _remove_output_dir(exp_id, record.output_dir)
            store.delete(exp_id)
            deleted += 1
            suffix = "（含输出文件）" if files_removed else "（仅元数据）"
            print(f"已删除: {exp_id} {suffix}")
    print(f"共删除 {deleted}/{len(experiment_ids)} 个实验")
