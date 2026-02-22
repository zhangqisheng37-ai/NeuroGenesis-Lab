# NeuroGenesis Lab

NeuroGenesis Lab 是一个基于 Python 的神经进化仿真框架，旨在模拟数字生命的演化、学习与神经网络动力学。

本项目结合了脉冲神经网络 (SNN)、演化算法与符号学习机制，提供了一个高度可配置、可观测的实验平台，用于研究人工生命的涌现行为、规则发现与网络自组织特性。

## 当前状态

- 阶段四（长程稳定 + 可视化 + 学习集成 + 持久化恢复）已交付。
- 技术债进度：TD-1、TD-2、TD-3、TD-4、TD-5 已完成，剩余 TD-6、TD-7。
- 最新测试基线：217 passed, 0 failed（2026-02-22）。

## 核心特性

- **生物启发模型**：基于能量约束的 LIF (Leaky Integrate-and-Fire) 神经元模型，支持动态突触与可塑性。
- **演化系统**：内置自然选择机制，包括基于适应度的繁殖、突变、死亡与尸体回收（Scavenging）。
- **空间动力学**：神经元具有二维空间坐标，支持基于空间距离的连接策略（Proximity Connection）与消耗能量的迁移（Migration）行为。
- **混合学习引擎**：结合符号规则发现 (Rule Discovery) 与神经动力学预测，支持从脉冲序列中提取因果逻辑，并与仿真流自动集成（AutoLearner）。
- **高度可观测**：全链路数据记录（JSONL），支持 tick 级的时间序列分析，并提供**实时可视化监控面板**（种群、能量、学习指标、空间拓扑）。
- **长程稳定性**：内置自适应能量衰减、能量上限与濒危保护机制，确保 500+ ticks 的长程演化稳定运行。
- **持久化与恢复**：支持按间隔自动保存 checkpoint，并从中断点恢复继续运行，适配长程实验。
- **输出治理**：支持日志采样（`--log-interval`）、压缩输出（`--gzip-output`）与按大小轮转（`--max-file-size`）。
- **高性能架构**：内置基于 NumPy + scipy.sparse 的向量化仿真内核 (`VectorizedNetwork`)，利用稀疏矩阵存储连接权重、向量化放电传播与空间网格索引，支持万级神经元规模的实时演化。
- **统一引擎接口**：`Network` 与 `VectorizedNetwork` 遵循 `SimEngine` 协议，返回统一的 `TickSnapshot` 和 `EngineSummary` 类型，上层代码可无缝切换引擎。

## 快速开始

### 安装

环境要求：Python >= 3.10

```bash
# 安装基础包
pip install -e .

# 安装包含可视化依赖的完整包
pip install -e ".[viz]"
```

### 运行实验

```bash
# 运行默认实验（100 ticks）
python -m neurogenesis_lab run

# 运行并开启实时可视化监控面板
python -m neurogenesis_lab run --visualize

# 使用自定义配置文件运行（如硬核挑战模式）
python -m neurogenesis_lab run --config experiments/challenge_hardcore.toml

# 长程实验输出治理（采样 + gzip + 轮转）
python -m neurogenesis_lab run --ticks 2000 --log-interval 10 --gzip-output --max-file-size 5000000

# 每 100 tick 自动保存 checkpoint
python -m neurogenesis_lab run --ticks 1000 --checkpoint-interval 100 --checkpoint-dir outputs/checkpoints

# 从 checkpoint 恢复，继续运行 200 tick
python -m neurogenesis_lab run --resume-from outputs/checkpoints/checkpoint_tick_1000.pkl --ticks 200

# 性能基准测试（例如 5000 神经元）
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100

# 查看版本与默认配置
python -m neurogenesis_lab info

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

运行测试（要求最低 75% 的测试覆盖率）：

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_network.py

# 可选：显式指定 pytest 临时目录（Windows 权限环境下推荐）
pytest --basetemp=.pytest_tmp

# 代码格式化与静态检查
ruff check src/
```

## 文档

项目文档位于 `docs/` 目录：

| 文档 | 说明 |
| ------ | ------ |
| [快速开始](docs/quickstart.md) | 安装、基本使用、代码示例 |
| [参数说明](docs/parameters.md) | 所有配置参数详解 |
| [架构设计](docs/architecture.md) | 系统架构与核心概念 |
| [可观测性指标](docs/metrics.md) | 输出数据字段说明 |
| [风险清单](docs/risk_list.md) | 当前已知风险与待办事项 |
| [更新日志](docs/CHANGELOG.md) | 版本变更历史 |
| [阶段四开发清单](docs/开发清单04.md) | 当前阶段交付与验收状态 |

## 开发路线图

项目的开发进度通过 `docs/` 目录下的开发清单进行管理：

- [x] [阶段一：基础脉冲与网络拓扑](docs/开发清单01已完成.md)
- [x] [阶段二：演化与空间动力学](docs/开发清单02 已完成.md)
- [x] [阶段三：符号学习与规则发现](docs/开发清单03_已完成.md)
- [x] [阶段四：长程稳定 + 可视化 + 学习集成 + 持久化恢复](docs/开发清单04.md)（已完成，技术债持续清理）
