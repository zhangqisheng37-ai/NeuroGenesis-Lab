# 教程 02：参数调优指南

NeuroGenesis Lab 提供了丰富的参数来控制神经演化过程。本教程将介绍如何通过修改 TOML 配置文件来调整实验。

## 1. 配置文件结构

配置文件分为四个主要部分：

- `[simulation]`：仿真全局参数（ticks, seed, dt）
- `[neuron]`：神经元模型参数（LIF 动力学）
- `[network]`：网络与演化参数（繁殖、变异、能量衰减）
- `[evolution]`：适应度权重与惩罚机制

## 2. 核心参数解析

### 2.1 能量经济

能量是神经元生存和繁殖的货币。

- `energy_decay` (默认 0.003)：每 tick 消耗的基础能量。
- `spike_cost` (默认 0.05)：每次放电消耗的能量。
- `reproduction_threshold` (默认 2.5)：繁殖所需的最小能量。
- `reproduction_share` (默认 0.4)：繁殖时分给子代的能量比例。

**调优建议**：
- 如果种群迅速崩溃，尝试降低 `energy_decay` 或提高 `input_current_scale`。
- 如果种群爆炸式增长，尝试提高 `reproduction_threshold` 或 `energy_decay`。

### 2.2 演化与变异

- `mutation_scale` (默认 0.05)：网络默认突变幅度。
- `mutation_amplitude` (默认 0.05)：演化阶段优先使用的突变幅度（会覆盖 `mutation_scale`）。
- `congestion_penalty` (默认 0.1)：高连接神经元的繁殖惩罚系数。

**调优建议**：
- 提高变异率可以加速探索，但也可能导致不稳定。

### 2.3 空间系统

- `scavenger_radius` (默认 2.5)：尸体回收有效半径。
- `max_links` (默认 200)：全网最大连接数。
- `move_cost` (默认 0.01)：神经元移动消耗的能量。

**调优建议**：
- 启用空间连接可以观察到聚类现象。降低 `move_cost` 鼓励神经元探索空间。

## 3. 预设配置模板

我们提供了几个预设配置，你可以直接使用或作为起点：

- `experiments/experiment_stable.toml`：长程稳定配置，启用了自适应衰减和繁殖机制。
- `experiments/experiment_high_mutation.toml`：高变异率配置，用于探索演化动力学。
- `experiments/experiment_spatial.toml`：强调空间结构的配置。
- `experiments/experiment_learning.toml`：学习增强配置，高预测奖励。

## 4. 批量实验

你可以使用 `batch_experiments.py` 工具同时运行多个配置，并生成对比报告：

```bash
python examples/batch_experiments.py --configs experiments/experiment_stable.toml experiments/experiment_spatial.toml --workers 2
```

这将在 `outputs/batch_results/` 目录下生成 `comparison_report.html`。

## 5. CLI 快速调参（不改 TOML）

当你只想快速验证方向时，可直接用 CLI 覆盖：

```bash
python -m neurogenesis_lab run --config experiments/experiment_stable.toml --ticks 500 --seed 42 --input-weight 60 --stim-interval 3 --visualize
```

高频日志建议：

```bash
python -m neurogenesis_lab run --config experiments/experiment_stable.toml --ticks 500 --async-log --log-buffer-size 512
```

## 6. 性能调优验证

若已启用 Rust 扩展，可使用以下命令检查并行微基准：

```bash
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100 --rust-size 10000
```

输出中 `Rust并行LIF (backend=rust)` 表示已走 Rust 路径；若为 `python-fallback`，请先运行：

```bash
python -m maturin develop --release
python -m neurogenesis_lab info
```
