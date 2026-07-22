#!/usr/bin/env bash
#
# 从干净状态一步搭建 NeuroGenesis Lab 开发环境（Linux/macOS）。
#
# 若 .venv 不存在则自动用系统 Python 创建，安装开发依赖，
# 随后（默认）构建 Rust 扩展并执行运行时一致性检查。
# 使用 --skip-rust 可仅搭建纯 Python 开发环境（无需 Rust 工具链）。
#
# 用法:
#   bash scripts/bootstrap.sh
#   bash scripts/bootstrap.sh --skip-rust
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

skip_rust=0
for arg in "$@"; do
  case "$arg" in
    --skip-rust) skip_rust=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

venv="$root/.venv"
python="$venv/bin/python"

if [ ! -x "$python" ]; then
  echo '[bootstrap] 未检测到 .venv，正在使用系统 Python 创建虚拟环境...'
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "$venv"
  elif command -v python >/dev/null 2>&1; then
    python -m venv "$venv"
  else
    echo '无法创建虚拟环境：未找到 python3/python。请先安装 Python 3.10+。' >&2
    exit 1
  fi
fi

echo '[bootstrap] 升级 pip...'
"$python" -m pip install --upgrade pip

echo '[bootstrap] 安装项目与开发依赖 (dev, viz)...'
"$python" -m pip install -e ".[dev,viz]"

if [ "$skip_rust" -eq 1 ]; then
  echo '[bootstrap] 已跳过 Rust 扩展构建与运行时检查 (--skip-rust)。'
  echo '[bootstrap] 完成：纯 Python 开发环境就绪。运行 pytest 即可验证。'
  exit 0
fi

echo '[bootstrap] 安装 Rust 构建依赖...'
"$python" -m pip install -e ".[rust]"

echo '[bootstrap] 构建并安装 Rust 扩展...'
"$python" -m maturin develop --release --manifest-path rust_src/neurogenesis_core/Cargo.toml

echo '[bootstrap] 检查 info 输出...'
"$python" -m neurogenesis_lab info

echo '[bootstrap] 执行发布前检查脚本...'
"$python" scripts/pre_release_check.py

echo '[bootstrap] 完成：Rust 运行时一致性检查通过。'
