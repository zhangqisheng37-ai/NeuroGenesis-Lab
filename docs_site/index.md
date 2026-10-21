# NeuroGenesis Lab

神经演化仿真框架 — 研究数字生命涌现、学习与神经网络动力学。

## 特性

- **脉冲神经网络 (SNN)**：LIF / Izhikevich / AdEx 多模型支持
- **演化算法**：自然选择、繁殖、变异、死亡、尸体回收
- **混合学习引擎**：符号规则发现 + 神经动力学预测（AutoLearner）
- **高性能**：向量化引擎 (NumPy + scipy.sparse) 支持万级神经元
- **Rust 加速**：PyO3 原生扩展，并行 LIF 计算与模式挖掘
- **双引擎对等**：`Network` 与 `VectorizedNetwork` 遵循统一 `SimEngine` 协议
- **实时可视化**：种群、能量、学习指标、空间拓扑实时监控面板
- **长程稳定**：自适应衰减/繁殖、濒危保护、能量上限机制
- **持久化与恢复**：checkpoint 自动保存与断点续跑
- **输出治理**：日志采样、gzip 压缩、文件轮转、异步双缓冲

## 快速安装

```bash
pip install -e .

# 含可视化依赖
pip install -e ".[viz]"

# Rust 扩展（可选）
pip install -e ".[rust]"
python -m maturin develop --release
```

## 运行仿真

```bash
# 默认实验
python -m neurogenesis_lab run --ticks 100

# 实时可视化
python -m neurogenesis_lab run --visualize

# 查看版本与 Rust 状态
python -m neurogenesis_lab info

# 性能基准
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100
```

## 当前状态

- **版本**：v0.6.1
- **测试基线**：全量测试通过，覆盖率门禁 ≥90%（CI 强制）
- **阶段**：Phase 8 进行中 — 实时可视化、实验管理、Web 管理界面已完成

> 实验产物目录约定见仓库根目录 `outputs/README.md`。

## 导航

- [快速开始](quickstart.md) — 安装、运行、代码示例
- [教程](tutorials/tutorial_01_first_experiment.md) — 从入门到高级
- [案例研究](case_studies/case_study_evolution.md) — 演化与学习实验分析
- [API 参考](api/config.md) — 完整模块文档
- [架构设计](architecture.md) — 系统内部工作原理
- [参数参考](parameters.md) — 所有可配置参数
- [指标说明](metrics.md) — 输出数据字段
- [更新日志](changelog.md) — 版本变更历史
