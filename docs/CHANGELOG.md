# 更新日志 (Changelog)

本文档记录 NeuroGenesis Lab 的所有重要版本变更。

---

## v0.3.0 (2026-02-15)

### 新增功能

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
|------|--------|
| `v_th` | -55.0 |
| `reproduction_threshold` | 2.5 |
| `energy_decay` | 0.003 |
| `min_neurons` | 2 |

### 测试

- 新增 174 个测试用例
- 测试覆盖率 92% (要求 75%)
- 添加 mypy 类型检查

---

## v0.2.0 (2026-01)

### 新增功能

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
