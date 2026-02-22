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

- `energy_decay` (默认 0.0015)：每 tick 消耗的基础能量。
- `spike_cost` (默认 0.04)：每次放电消耗的能量。
- `reproduction_threshold` (默认 1.8)：繁殖所需的最小能量。
- `reproduction_share` (默认 0.35)：繁殖时分给子代的能量比例。

**调优建议**：
- 如果种群迅速崩溃，尝试降低 `energy_decay` 或提高 `input_current_scale`。
- 如果种群爆炸式增长，尝试提高 `reproduction_threshold` 或 `energy_decay`。

### 2.2 演化与变异

- `mutation_scale` (默认 0.05)：变异发生的概率。
- `mutation_amplitude` (默认 0.05)：变异的幅度（影响神经元参数）。

**调优建议**：
- 提高变异率可以加速探索，但也可能导致不稳定。

### 2.3 空间系统

- `spatial_radius` (默认 5.0)：神经元连接的最大距离。
- `max_links_per_neuron` (默认 8)：每个神经元的最大连接数。
- `move_cost` (默认 0.008)：神经元移动消耗的能量。

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
