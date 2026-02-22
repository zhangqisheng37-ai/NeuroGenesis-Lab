# 教程 01：10 分钟快速上手

欢迎来到 NeuroGenesis Lab！本教程将带你快速运行第一个神经演化实验。

## 1. 安装与准备

首先，确保你已经安装了项目：

```bash
# 在项目根目录下运行
pip install -e .
```

如果你想使用实时可视化功能，请安装可选依赖：

```bash
pip install -e ".[viz]"
```

## 2. 运行最小实验

NeuroGenesis Lab 提供了一个最小配置，适合快速验证环境。

```bash
python -m neurogenesis_lab run --config experiments/experiment_minimal.toml
```

你将看到类似如下的输出：

```
[NeuroGenesis Lab v0.4.1] 运行仿真
  ticks=100, seed=42, schema=1.0
  tick 20/100 (global=20): alive=8, PE=0.0000, rules=0, acc=0.000
  tick 40/100 (global=40): alive=8, PE=0.0000, rules=0, acc=0.000
...
完成! 用时 0.15s
  输出: outputs\run_20260222_123456_789
  存活: 8/8
```

## 3. 启用实时可视化

为了更直观地观察神经元的演化，我们可以启用实时监控面板：

```bash
python -m neurogenesis_lab run --config experiments/experiment_minimal.toml --visualize
```

这会弹出一个窗口，实时显示：
1. **种群与能量曲线**：神经元数量和平均能量的变化。
2. **能量分布**：当前存活神经元的能量直方图。
3. **空间拓扑**：神经元在 2D 空间中的位置和连接。
4. **学习指标**：预测准确率和发现的规则数量。

## 4. 离线分析实验数据

实验结束后，所有数据都保存在 `outputs/` 目录下。你可以使用分析工具生成完整的 HTML 报告：

```bash
# 替换为你的实际输出路径
python examples/analyze_experiment.py --path outputs/run_20260222_123456_789/run.jsonl --show
```

## 5. 下一步

恭喜！你已经成功运行了第一个实验。接下来，你可以：
- 阅读 [教程 02：参数调优指南](tutorial_02_parameter_tuning.md) 了解如何控制演化过程。
- 尝试运行长程稳定实验：`python -m neurogenesis_lab run --config experiments/experiment_stable.toml`
