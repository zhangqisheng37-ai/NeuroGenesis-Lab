# NeuroGenesis Lab

NeuroGenesis Lab 是一个基于 Python 的神经进化仿真框架，旨在模拟数字生命的演化、学习与神经网络动力学。

本项目结合了脉冲神经网络 (SNN)、演化算法与符号学习机制，提供了一个高度可配置、可观测的实验平台，用于研究人工生命的涌现行为、规则发现与网络自组织特性。

## 核心特性

- **生物启发模型**：基于能量约束的 LIF (Leaky Integrate-and-Fire) 神经元模型，支持动态突触与可塑性。
- **演化系统**：内置自然选择机制，包括基于适应度的繁殖、突变、死亡与尸体回收（Scavenging）。
- **混合学习引擎**：结合符号规则发现 (Rule Discovery) 与神经动力学预测，支持从脉冲序列中提取因果逻辑。
- **高度可观测**：全链路数据记录（JSONL），支持 tick 级的时间序列分析与可视化。
- **高性能架构**：支持基于 NumPy 的向量化计算加速，可扩展至数千神经元规模。

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

# 查看帮助
python -m neurogenesis_lab --help
```

### 运行示例脚本

```bash
# 运行演化实验（观察种群波动）
python examples/experiment_evolution.py

# 运行对照实验（高变异 vs 低变异）
python examples/experiment_contrast.py
```

## 文档索引

更多详细文档请参考 `docs/` 目录：

- [快速入门 (Quickstart)](docs/quickstart.md): 详细的安装与上手指南。
- [参数说明 (Parameters)](docs/parameters.md): 仿真参数与配置文件详解。
- [架构设计 (Architecture)](docs/architecture.md): 系统核心架构与设计原理（待补充）。
- [常见问题 (FAQ)](docs/faq.md): 常见问题解答。

## 项目结构

```
src/neurogenesis_lab/
├── config.py          # 配置系统
├── neuron.py          # 神经元模型
├── network.py         # 网络演化引擎
├── learning.py        # 学习与规则发现
├── encoding.py        # 符号编码系统
└── observability.py   # 数据采集与监控
```

## 开发

运行测试：

```bash
pytest
```
