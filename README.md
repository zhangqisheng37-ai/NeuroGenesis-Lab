# NeuroGenesis Lab

NeuroGenesis Lab 是一个基于 Python 的神经进化仿真框架，旨在模拟数字生命的演化、学习与神经网络动力学。

本项目结合了脉冲神经网络 (SNN)、演化算法与符号学习机制，提供了一个高度可配置、可观测的实验平台，用于研究人工生命的涌现行为、规则发现与网络自组织特性。

## 核心特性

- **生物启发模型**：基于能量约束的 LIF (Leaky Integrate-and-Fire) 神经元模型，支持动态突触与可塑性。
- **演化系统**：内置自然选择机制，包括基于适应度的繁殖、突变、死亡与尸体回收（Scavenging）。
- **混合学习引擎**：结合符号规则发现 (Rule Discovery) 与神经动力学预测，支持从脉冲序列中提取因果逻辑。
- **高度可观测**：全链路数据记录（JSONL），支持 tick 级的时间序列分析与可视化。
- **输出治理**：支持日志采样（`--log-interval`）、压缩输出（`--gzip-output`）与按大小轮转（`--max-file-size`）。
- **高性能架构**：内置基于 NumPy 的向量化仿真内核 (`VectorizedNetwork`)，利用空间网格索引与矩阵运算，支持万级神经元规模的实时演化。

## 快速开始

### 安装

环境要求：Python >= 3.10

```bash
pip install -e .
```

### 运行最小实验

```bash
# 运行默认实验（100 ticks）
python -m neurogenesis_lab run

# 长程实验输出治理（采样 + gzip + 轮转）
python -m neurogenesis_lab run --ticks 2000 --log-interval 10 --gzip-output --max-file-size 5000000

# 查看帮助
python -m neurogenesis_lab --help
```

### 运行示例脚本

```bash
# 运行演化实验（观察种群波动）
python examples/experiment_evolution.py

# 运行对照实验（高变异 vs 低变异）
python examples/experiment_contrast.py

# 批量解压 .jsonl.gz 输出文件
python examples/decompress_outputs.py --path outputs --recursive
```

## 开发

运行测试：

```bash
pytest
```
