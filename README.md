# NeuroGenesis Lab

NeuroGenesis Lab 是一个基于 Python 的神经进化仿真框架，用于研究数字生命的演化、学习与网络动力学。

## 当前状态

- 当前版本：`v0.6.0`（`src/neurogenesis_lab/__init__.py`）
- 当前 Schema：`2.0.0`（`src/neurogenesis_lab/config.py`）
- CLI 子命令：`run` / `benchmark` / `info`
- 双引擎统一协议：`Network` 与 `VectorizedNetwork` 均实现 `SimEngine`（`TickSnapshot` / `EngineSummary`）

## 核心能力

- 多模型神经元动力学：LIF / Izhikevich / AdEx
- 演化机制：繁殖、突变、死亡、尸体回收、濒危保护
- 学习机制：`AutoLearner` + `RuleDiscovery`
- 输出治理：采样、gzip 压缩、按文件大小轮转
- 异步日志：双缓冲写盘（降低主循环 I/O 阻塞）
- checkpoint：保存与恢复（`--checkpoint-interval` / `--resume-from`）
- Rust 桥接：`core_bridge.py` 提供可用性探测与 Python fallback

## 安装

环境要求：Python >= 3.10

```bash
# 基础安装
pip install -e .

# 可视化依赖（可选）
pip install -e ".[viz]"

# Rust 依赖（可选）
pip install -e ".[rust]"
python -m maturin develop --release
```

## 快速开始

```bash
# 查看总帮助
python -m neurogenesis_lab --help

# run 参数帮助
python -m neurogenesis_lab run --help

# benchmark 参数帮助
python -m neurogenesis_lab benchmark --help

# 版本 / Schema / Rust 状态 / 默认配置
python -m neurogenesis_lab info

# 默认运行（ticks 默认 100）
python -m neurogenesis_lab run

# 指定运行参数
python -m neurogenesis_lab run --ticks 200 --seed 123 --neurons 12

# 使用 TOML 配置
python -m neurogenesis_lab run --config config.toml

# 输出治理：采样 + gzip + 轮转
python -m neurogenesis_lab run --ticks 500 --log-interval 10 --gzip-output --max-file-size 5000000

# 异步双缓冲日志
python -m neurogenesis_lab run --ticks 500 --async-log --log-buffer-size 512

# checkpoint：每 100 个“本次运行 tick”保存一次
python -m neurogenesis_lab run --ticks 1000 --checkpoint-interval 100 --checkpoint-dir outputs/checkpoints

# 从 checkpoint 恢复后继续运行 200 tick
python -m neurogenesis_lab run --resume-from outputs/checkpoints/checkpoint_tick_1000.pkl --ticks 200

# 性能基准
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100 --rust-size 10000
```

## 输出说明

`run` 默认会在 `outputs/` 下创建以时间戳命名的实验目录，并写入：

- `run.jsonl` 或 `run.jsonl.gz`（启用 `--gzip-output` 时）
- 轮转后文件形如：`run_0001.jsonl(.gz)`、`run_0002.jsonl(.gz)`
- checkpoint 文件形如：`checkpoint_tick_<global_tick>.pkl`

## 项目结构

```text
src/neurogenesis_lab/
├── __init__.py
├── __main__.py
├── config.py
├── neuron.py
├── network.py
├── vectorized.py
├── protocols.py
├── learning.py
├── encoding.py
├── observability.py
├── core_bridge.py
├── visualization.py
└── cli/
    ├── __init__.py
    ├── parser.py
    ├── config_builder.py
    ├── run.py
    ├── benchmark.py
    └── info.py
```

## 实验配置索引

`experiments/` 目录包含可直接复用的配置：

- 基础：`experiment_minimal.toml`、`experiment_stable.toml`、`experiment_optimized.toml`
- 对照/专题：`experiment_contrast.toml`、`experiment_spatial.toml`、`experiment_learning.toml`
- 挑战：`challenge_hardcore.toml`、`challenge_logic.toml`、`challenge_war.toml`
- 消融：`ablation_adaptive_on.toml`、`ablation_adaptive_off.toml`

## 开发

```bash
# 全量测试
pytest

# 指定测试
pytest tests/test_network.py

# Lint
ruff check src/
```

## 文档

- [快速开始](docs/quickstart.md)
- [参数说明](docs/parameters.md)
- [架构设计](docs/architecture.md)
- [可观测性指标](docs/metrics.md)
- [风险清单](docs/risk_list.md)
- [更新日志](docs/CHANGELOG.md)
- [教程 01](docs/tutorial_01_first_experiment.md)
- [教程 02](docs/tutorial_02_parameter_tuning.md)
- [教程 03](docs/tutorial_03_custom_neuron.md)
- [案例：演化动力学](docs/case_study_evolution.md)
- [案例：规则涌现](docs/case_study_learning.md)
