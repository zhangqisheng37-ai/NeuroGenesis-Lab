<#
.SYNOPSIS
    从干净状态一步搭建 NeuroGenesis Lab 开发环境。

.DESCRIPTION
    若 .venv 不存在则自动用系统 Python 创建，安装开发依赖，
    随后（默认）构建 Rust 扩展并执行运行时一致性检查。
    使用 -SkipRust 可仅搭建纯 Python 开发环境（无需 Rust 工具链）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -SkipRust
#>
[CmdletBinding()]
param(
    [switch]$SkipRust
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([Parameter(Mandatory)][scriptblock]$Command, [string]$What)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "步骤失败（退出码 $LASTEXITCODE）：$What"
    }
}

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venv = Join-Path $root '.venv'
$python = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host '[bootstrap] 未检测到 .venv，正在使用系统 Python 创建虚拟环境...'
    $created = $false
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $venv
        if ($LASTEXITCODE -eq 0) { $created = $true }
    }
    if (-not $created -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -m venv $venv
        if ($LASTEXITCODE -eq 0) { $created = $true }
    }
    if (-not $created) {
        throw '无法创建虚拟环境：未找到可用的系统 Python（尝试过 py -3 与 python）。请先安装 Python 3.10+。'
    }
}

Write-Host '[bootstrap] 升级 pip...'
Invoke-Checked { & $python -m pip install --upgrade pip } 'pip 升级'

Write-Host '[bootstrap] 安装项目与开发依赖 (dev, viz)...'
Invoke-Checked { & $python -m pip install -e ".[dev,viz]" } '安装开发依赖'

if ($SkipRust) {
    Write-Host '[bootstrap] 已跳过 Rust 扩展构建与运行时检查 (-SkipRust)。'
    Write-Host '[bootstrap] 完成：纯 Python 开发环境就绪。运行 pytest 即可验证。'
    return
}

Write-Host '[bootstrap] 安装 Rust 构建依赖...'
Invoke-Checked { & $python -m pip install -e ".[rust]" } '安装 Rust 构建依赖'

Write-Host '[bootstrap] 构建并安装 Rust 扩展...'
Invoke-Checked { & $python -m maturin develop --release --manifest-path rust_src/neurogenesis_core/Cargo.toml } '构建 Rust 扩展'

Write-Host '[bootstrap] 检查 info 输出...'
Invoke-Checked { & $python -m neurogenesis_lab info } 'info 输出检查'

Write-Host '[bootstrap] 执行发布前检查脚本...'
Invoke-Checked { & $python scripts/pre_release_check.py } '发布前检查'

Write-Host '[bootstrap] 完成：Rust 运行时一致性检查通过。'
