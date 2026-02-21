"""离线实验分析工具 — 从 JSONL 生成可视化报告"""

import argparse
from pathlib import Path

from neurogenesis_lab.visualization import generate_report, VisualizationError


def main():
    parser = argparse.ArgumentParser(
        description="NeuroGenesis Lab 实验分析工具"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="JSONL 数据文件路径（支持 .jsonl 或 .jsonl.gz）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录（默认为数据文件同目录下的 analysis/）"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="显示图表（而非仅保存）"
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.path)
    
    if not data_path.exists():
        print(f"错误: 文件不存在: {data_path}")
        return 1
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = data_path.parent / "analysis"
    
    try:
        generate_report(data_path, output_dir, show=args.show)
        print(f"\n✅ 分析完成！")
        print(f"   输出目录: {output_dir}")
        return 0
    except VisualizationError as e:
        print(f"错误: {e}")
        return 1
    except Exception as e:
        print(f"未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

