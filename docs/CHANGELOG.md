# 更新日志 (Changelog)

本文档记录 NeuroGenesis Lab 的所有重要版本变更。

---

## Unreleased (2026-02-22)

### 技术债务与测试补齐

- **TD-1：演化流程解耦**
  - `Network.tick()` 内演化逻辑提取为 `_run_evolution_step()`。
  - `check_reproduction()` 与 `check_death()` 改为返回增量计数，便于独立测试和复用。
- **TD-3：`Neuron.observe()` 性能优化**
  - 增加 `transition_counts` 缓存与 dirty-flag，仅在 transitions 变更时重建观察结果。
- **TD-5：CLI 冒烟测试补齐**
  - 新增从 `main()` 入口执行 `run` 与 `benchmark` 的链路冒烟测试。
  - `run` 冒烟测试增加 checkpoint 文件输出校验。

### 验证

- 最新全量测试：217 passed, 0 failed。

---

## v0.4.1 (2026-02-22)

### v0.4.1 新增功能

- **持久化与恢复（Checkpoint）**
  - `Network` 新增 `save_checkpoint(path)` 与 `load_checkpoint(path)`。
  - checkpoint 保存网络状态、事件队列、尸体列表与随机数状态。
  - 增加 schema 主版本兼容校验，避免不兼容状态文件误加载。
- **CLI checkpoint 控制**
  - `run` 命令新增 `--checkpoint-interval`、`--checkpoint-dir`、`--resume-from`。
  - 支持从全局 tick 断点续跑，恢复后可继续记录与可视化。

### v0.4.1 测试与工程改进

- 新增 `tests/test_checkpoint.py`，覆盖：
  - 保存/加载 roundtrip
  - schema 不兼容拒绝
  - 连续运行 vs 中断恢复一致性
- 补充 CLI 参数解析测试（checkpoint 参数）。
- pytest 配置增加 `--basetemp=.pytest_tmp`，规避部分 Windows 环境系统临时目录权限问题。

---

## v0.4.0 (2026-02-22)

### v0.4.0 新增功能

- **长程稳定性保障**
  - 引入自适应能量衰减机制：根据种群密度动态调整 `energy_decay`。
  - 引入自适应繁殖成本：根据种群规模动态调整 `reproduction_share`。
  - 增加能量上限 `energy_cap`，防止单个神经元无限积累能量。
  - 增强濒危保护模式：种群数量低于阈值时，能量衰减减半。
- **实时可视化系统**
  - 新增 `visualization.py` 模块，支持使用 `matplotlib` 绘制种群、能量、学习指标和空间拓扑图。
  - CLI 增加 `--visualize` 参数，启动实时监控面板。
  - 提供 `examples/analyze_experiment.py` 离线分析脚本，生成完整报告。
- **学习引擎深度集成**
  - 新增 `AutoLearner` 包装类，自动从仿真流中提取样本并更新规则。
  - 引入预测驱动奖励：神经元预测正确时获得能量奖励，错误时受到轻微惩罚。
  - 规则库增加容量上限和淘汰机制。
- **空间系统激活**
  - `Network` 增加 `connect_by_proximity()` 方法，支持基于空间距离建立连接。
  - 神经元增加 `move()` 方法，支持消耗能量进行空间迁移。
  - 繁殖时子代位置在母体附近（高斯扰动）。

### v0.4.0 依赖

- 新增可选依赖 `matplotlib>=3.5`，可通过 `pip install -e ".[viz]"` 安装。

---

## v0.3.1 (2026-02-20)

### 双引擎接口统一

- 新增 `protocols.py`，定义 `TickSnapshot`、`EngineSummary`、`SimEngine` 协议
- `Network.tick()` 返回字段变更：`emitted_events` → `propagation_count`，新增 `mean_energy`
- `Network.get_summary()` 返回字段变更：`avg_energy` → `mean_energy`，移除 `corpses` 字段
- `VectorizedNetwork.get_summary()` 新增 `total_propagations`、`active_ratio` 字段
- 删除旧的 `SnapshotDict`、`NetworkSummaryDict`、`VectorizedSummaryDict` 类型定义
- 上层代码（observability、visualize_evolution、examples）全部适配新字段名

### 性能优化

- `VectorizedNetwork` 连接矩阵从 N×N 密集 ndarray 替换为 `scipy.sparse.csr_matrix`，万级神经元内存占用从 ~800MB 降至数 MB
- `connect_radius` 使用 `scipy.spatial.distance.cdist` 向量化距离计算
- 放电传播使用稀疏矩阵行切片 `nnz` 替代 Python for 循环
- `get_spike_input_currents` 使用稀疏矩阵 `sum(axis=0)` 向量化
- 繁殖时通过 `lil_matrix` 局部修改再转回 `csr_matrix`，避免频繁格式转换
- `Network._update_corpses` 引入网格索引优化近邻搜索，从 O(corpses × alive) 降至 O(corpses × 9 cells)

### v0.3.1 依赖

- 新增 `scipy>=1.10` 依赖

### v0.3.1 测试

- 新增 175 个测试用例（+1）
- 测试覆盖率 92.72%

---

## v0.3.0 (2026-02-15)

### v0.3.0 新增功能

- **输出治理系统**
  - 支持 `--log-interval` 日志采样间隔
  - 支持 `--gzip-output` 压缩输出
  - 支持 `--max-file-size` 文件自动轮转
  - 提供 `examples/decompress_outputs.py` 批量解压脚本

- **向量化引擎 (VectorizedNetwork)**
  - 基于 NumPy 的高性能仿真内核
  - 支持万级神经元规模
  - 提供 `SpatialGrid` 空间索引加速近邻查找
  - 性能提升 10-50 倍

- **CLI 增强**
  - `info` 命令：显示版本和 schema 版本
  - `benchmark` 命令：性能基准测试
  - TOML 配置文件完整支持

### 参数优化（config.toml 中的优化配置）

> 注：以下为推荐配置，非代码默认值。默认值见 `config.py`。

- 降低神经元放电阈值 (`v_th: -55.0 → -50.0`)
- 降低繁殖门槛 (`reproduction_threshold: 2.5 → 1.8`)
- 减半能量衰减 (`energy_decay: 0.003 → 0.0015`)
- 提高最小神经元保护 (`min_neurons: 2 → 5`)

### 代码默认值

| 参数 | 默认值 |
| ------ | -------- |
| `v_th` | -55.0 |
| `reproduction_threshold` | 2.5 |
| `energy_decay` | 0.003 |
| `min_neurons` | 2 |

### v0.3.0 测试

- 新增 174 个测试用例
- 测试覆盖率 92% (要求 75%)
- 添加 mypy 类型检查

---

## v0.2.0 (2026-01)

### v0.2.0 新增功能

- **学习引擎**
  - `PatternMiner`: 表达式模式挖掘
  - `RuleDiscovery`: 基于 trigram/bigram 的规则发现
  - `RuleComposer`: 规则组合与缺失 token 填充
  - `QueryEngine`: 表达式查询接口

- **可观测性系统**
  - `NetworkObserver`: 网络级别指标收集
  - `ObservabilityLogger`: 实验日志记录
  - 支持 `debug_neuron_ids` 追踪特定神经元

### 神经元模型

- `LIFNeuron`: Leaky Integrate-and-Fire 模型
- `IzhikevichNeuron`: Izhikevich 神经元模型
- `AdExNeuron`: AdEx 指数整合放电模型

---

## v0.1.0 (2025-12)

### 初始版本

- 基础仿真框架
- 事件驱动网络引擎
- 演化机制（繁殖、死亡、尸体回收）
- 基本 CLI 接口

---

## 版本语义

版本号遵循 [语义化版本](https://semver.org/) 规范：

- **主版本 (MAJOR)**: 不兼容的 API 变更
- **次版本 (MINOR):**: 向后兼容的新功能
- **补丁版本 (PATCH)**: 向后兼容的 bug 修复

---

## 文档

- [快速开始 (Quickstart)](quickstart.md)
- [参数说明 (Parameters)](parameters.md)
- [架构设计 (Architecture)](architecture.md)
- [可观测性指标 (Metrics)](metrics.md)
- [风险清单 (Risk List)](risk_list.md)
