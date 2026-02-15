"""批量解压 .jsonl.gz 输出文件。

用法示例：
    # 解压某次实验目录
    python examples/decompress_outputs.py --path outputs/run_20260215_095221

    # 递归解压整个 outputs 目录
    python examples/decompress_outputs.py --path outputs --recursive

    # 只预览将要解压的文件，不实际写入
    python examples/decompress_outputs.py --path outputs --recursive --dry-run

    # 即使 .jsonl 已存在也覆盖
    python examples/decompress_outputs.py --path outputs --recursive --overwrite
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量解压 .jsonl.gz 文件为 .jsonl")
    parser.add_argument(
        "--path",
        type=str,
        default="outputs",
        help="目标目录或单个 .jsonl.gz 文件路径（默认: outputs）",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归搜索子目录中的 .jsonl.gz 文件",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标 .jsonl 已存在则覆盖",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划，不执行解压",
    )
    return parser.parse_args()


def find_gzip_files(target: Path, recursive: bool) -> list[Path]:
    if target.is_file():
        return [target] if target.name.endswith(".jsonl.gz") else []
    if not target.exists() or not target.is_dir():
        return []
    pattern = "**/*.jsonl.gz" if recursive else "*.jsonl.gz"
    return sorted(target.glob(pattern))


def output_path_for(gz_path: Path) -> Path:
    if gz_path.name.endswith(".jsonl.gz"):
        return gz_path.with_name(gz_path.name[:-3])
    return gz_path.with_suffix("")


def decompress_one(gz_path: Path, out_path: Path, overwrite: bool, dry_run: bool) -> str:
    if dry_run:
        if out_path.exists() and not overwrite:
            return f"plan-skip {gz_path} -> {out_path} (exists)"
        return f"plan      {gz_path} -> {out_path}"

    if out_path.exists() and not overwrite:
        return f"skip   {gz_path} -> {out_path} (exists)"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rt", encoding="utf-8") as src:
        content = src.read()
    out_path.write_text(content, encoding="utf-8")
    return f"ok     {gz_path} -> {out_path}"


def main() -> None:
    args = parse_args()
    target = Path(args.path)

    files = find_gzip_files(target, recursive=args.recursive)
    if not files:
        print("未找到 .jsonl.gz 文件。")
        return

    done = 0
    skipped = 0
    planned = 0

    for gz_path in files:
        out_path = output_path_for(gz_path)
        message = decompress_one(
            gz_path,
            out_path,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        print(message)
        if message.startswith("ok"):
            done += 1
        elif message.startswith("skip"):
            skipped += 1
        elif message.startswith("plan"):
            planned += 1

    if args.dry_run:
        print(f"\n完成预览: plan={planned}, files={len(files)}")
    else:
        print(f"\n完成: decompressed={done}, skipped={skipped}, total={len(files)}")


if __name__ == "__main__":
    main()
