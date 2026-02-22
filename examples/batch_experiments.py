"""批量实验运行工具 — 支持参数扫描和并行执行"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime


def run_experiment(
    config_path: Path, output_base_dir: Path, label: str, description: str
) -> dict:
    """运行单个实验并返回结果元数据"""
    print(f"🚀 开始实验: {label} ({config_path.name})")
    start_time = time.time()

    # 运行 CLI 命令
    cmd = [
        "python",
        "-m",
        "neurogenesis_lab",
        "run",
        "--config",
        str(config_path),
        "--output",
        str(output_base_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    # 解析输出目录
    output_dir = None
    for line in result.stdout.splitlines():
        if line.strip().startswith("输出:"):
            output_dir = Path(line.split("输出:")[1].strip()).parent
            break

    success = result.returncode == 0

    if not success:
        print(f"❌ 实验失败: {label}")
        print(result.stderr)
        return {
            "label": label,
            "config": config_path.name,
            "success": False,
            "error": result.stderr,
        }

    print(f"✅ 实验完成: {label} (用时 {end_time - start_time:.2f}s)")

    # 尝试获取 git commit
    git_commit = "unknown"
    try:
        git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        if git_result.returncode == 0:
            git_commit = git_result.stdout.strip()
    except Exception:
        pass

    # 保存 metadata.json
    metadata = {
        "label": label,
        "description": description,
        "config_file": config_path.name,
        "git_commit": git_commit,
        "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": end_time - start_time,
        "status": "success",
    }

    if output_dir and output_dir.exists():
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def generate_comparison_report(results: list[dict], output_dir: Path):
    """生成简单的 HTML 对比报告"""
    html_content = [
        "<html><head><title>实验对比报告</title>",
        "<style>body{font-family:sans-serif;margin:20px;} table{border-collapse:collapse;width:100%;} th,td{border:1px solid #ddd;padding:8px;text-align:left;} th{background-color:#f2f2f2;}</style>",
        "</head><body>",
        "<h1>NeuroGenesis Lab 批量实验对比报告</h1>",
        "<table>",
        "<tr><th>标签</th><th>配置文件</th><th>状态</th><th>用时 (s)</th><th>描述</th></tr>",
    ]

    for res in results:
        status = "✅ 成功" if res.get("success", True) else "❌ 失败"
        duration = (
            f"{res.get('duration_seconds', 0):.2f}" if res.get("success", True) else "-"
        )
        html_content.append(
            f"<tr><td>{res.get('label')}</td><td>{res.get('config_file', res.get('config'))}</td>"
            f"<td>{status}</td><td>{duration}</td><td>{res.get('description', '')}</td></tr>"
        )

    html_content.append("</table></body></html>")

    report_path = output_dir / "comparison_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_content))

    print(f"\n📊 对比报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="批量运行实验")
    parser.add_argument(
        "--configs", nargs="+", required=True, help="要运行的配置文件列表"
    )
    parser.add_argument(
        "--output-dir", default="outputs/batch_results", help="输出基础目录"
    )
    parser.add_argument("--workers", type=int, default=2, help="并行工作进程数")

    args = parser.parse_args()

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    configs = [Path(c) for c in args.configs]

    print(f"准备运行 {len(configs)} 个实验，最大并行数: {args.workers}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, config_path in enumerate(configs):
            if not config_path.exists():
                print(f"⚠️ 警告: 配置文件不存在 {config_path}")
                continue

            label = f"exp_{i+1}_{config_path.stem}"
            description = f"Batch run for {config_path.name}"

            future = executor.submit(
                run_experiment, config_path, output_base_dir, label, description
            )
            futures.append(future)

        for future in as_completed(futures):
            results.append(future.result())

    generate_comparison_report(results, output_base_dir)


if __name__ == "__main__":
    main()
