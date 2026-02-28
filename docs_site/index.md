# NeuroGenesis Lab

神经演化仿真框架 — 研究数字生命涌现、学习与神经网络动力学。

## 特性

- **脉冲神经网络 (SNN)**：LIF / Izhikevich / AdEx 多模型支持
- **演化算法**：自然选择、繁殖、变异、死亡、尸体回收
- **混合学习引擎**：符号规则发现 + 神经动力学预测
- **高性能**：向量化引擎 (NumPy + scipy.sparse) 支持万级神经元
- **Rust 加速**：PyO3 原生扩展，并行 LIF 计算

## 快速安装

```bash
pip install -e .
```

## 运行仿真

```bash
python -m neurogenesis_lab run --ticks 100
```

详见 [快速开始](quickstart.md) 和 [教程](tutorials/tutorial_01_first_experiment.md)。
