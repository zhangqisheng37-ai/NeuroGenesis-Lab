$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "未找到 Python 虚拟环境: $python。请先创建并激活 .venv"
}

Write-Host '[bootstrap] 安装项目与 Rust 依赖...'
& $python -m pip install -e ".[rust]"

Write-Host '[bootstrap] 构建并安装 Rust 扩展...'
& $python -m maturin develop --release --manifest-path rust_src/neurogenesis_core/Cargo.toml

Write-Host '[bootstrap] 检查 info 输出...'
& $python -m neurogenesis_lab info

Write-Host '[bootstrap] 执行发布前检查脚本...'
& $python scripts/pre_release_check.py

Write-Host '[bootstrap] 完成：Rust 运行时一致性检查通过。'
