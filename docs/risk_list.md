# 下一阶段风险清单（v0.6.x）

> 更新时间：2026-02-28
>
> 本清单基于阶段七当前实现状态：VectorizedNetwork 功能对齐（多模型/尸体回收/自适应/checkpoint）、CLI 子包拆分、CI/CD 管道、API 文档生成已完成。代码基线 v0.6.0，347 tests，92.61% 覆盖率。

---

## 🔴 高优先级风险（仍需处理）✅ 已修复/已缓解

### R1: Rust 扩展运行时一致性依赖本地环境

**现象**: 在部分机器上可能出现“可构建但未加载”或 PATH/工具链缺失。
**影响**: 基准会回落到 `python-fallback`，导致性能评估偏差。
**建议**:

1. 发布前固定执行 `python -m neurogenesis_lab info`，确认 `Rust Core: available`
2. 将 `maturin develop --release` 纳入本地初始化脚本
3. CI 增加“导入扩展 + 组件探测”步骤

### R2: 激活动力学对外部输入强依赖 ✅ 已修复/已缓解

**状态**: 已落地背景激活与告警阈值配置化。
**进展**: 已新增并接入 `background_current`、`spontaneous_spike_rate`、`low_activation_threshold` 参数（配置/TOML/CLI/运行时均可用）。
**建议**: 持续补充“激活态配置模板”，并在长程实验中跟踪低激活告警触发率。

### R3: 能量经济仍可能出现双极化（爆发/衰亡） ✅ 已缓解

**状态**: v0.4.0 已引入自适应衰减、能量上限和濒危保护机制，500+ ticks 实验已能稳定运行。
**建议**: 持续观察更长程（2000+ ticks）实验中的能量分布。

### R4: 向量化与原始引擎已实现功能对齐 ✅ 已阶段性缓解

**现象**: 已建立统计可比测试，但模型细节尚未完全对齐。
**进展**: v0.6.0 已完成双引擎接口统一（`TickSnapshot` / `EngineSummary` / `SimEngine` 协议），`VectorizedNetwork` 新增多模型动力学（LIF/Izh/AdEx）、尸体回收、自适应衰减/繁殖、激危保护、能量上限、checkpoint 持久化。
**待完成**: `AutoLearner` 集成、同 seed 长程对照测试。
**建议**:

1. 完成 P2.4 AutoLearner 向量化集成
2. 在 benchmark 中持续标注”可比项/不可比项”
3. 增加更多趋势对照（PE、活跃率、能量分布）

---

## 🟡 中优先级风险

### R5: 配置参数存在“已解析未消费”字段 ✅ 已修复

**进展**: 已建立参数消费矩阵，并补齐 `mutation_amplitude`/`energy_budget_per_tick` 的运行时消费映射。

### R6: 学习引擎与仿真引擎耦合度低 ✅ 已修复

**状态**: v0.4.0 已实现 `AutoLearner` 包装类，自动从仿真流中提取样本并更新规则，并提供预测奖励。

### R7: CLI 关键路径覆盖 ✅ 已修复

**状态**: 已补齐全部 CLI 冒烟测试。
**进展**: Phase 7 将 CLI 拆分为 `cli/` 子包（6 个模块），覆盖率进一步提升。

### R8: AutoLearner 未集成到 VectorizedNetwork

**现象**: `VectorizedNetwork` 已具备尸体回收、多模型、自适应等功能，但尚未接入 `AutoLearner` 学习引擎。
**影响**: 向量化引擎缺少规则发现与预测奖励能力。
**建议**: 作为 Phase 7 P2.4 待完成任务跟踪。

### R9: mypy 严格模式未零错误

**现象**: `mypy` 已集成到 lint CI，但严格模式下可能仍有警告。
**建议**: 分模块逐步启用严格模式，先完成 `protocols` / `config` / `encoding`，再推进核心模块。

---

## 🟢 低优先级风险

### R10: 无持久化/恢复功能 ✅ 已修复

**状态**: `Network` 和 `VectorizedNetwork` 均已支持 checkpoint 保存/恢复，含 schema 兼容校验。

### R11: 空间系统未使用 ✅ 已修复

**状态**: v0.4.0 已在 `Network` 中添加 `connect_by_proximity()` 方法，并支持基于能量的迁移行为。

### R12: 无可视化 ✅ 已修复

**状态**: v0.4.0 已添加 `visualization.py`，支持实时监控面板和离线报告生成。

---

## 技术债务

| 编号 | 描述 | 影响 |
| ------ | ------ | ------ |
| TD-6 | 配置参数“已解析未消费”字段 | ✅ 已修复 |
| TD-7 | 向量化引擎不支持复杂事件路由 | ✅ 已修复（v0.6.0 向量化引擎已支持事件注入） |
| TD-8 | VectorizedNetwork 未集成 AutoLearner | 待 Phase 7 P2.4 完成 |
| TD-9 | mypy strict 零错误 | 待 Phase 7 P6.6 完成 |

---

## 已完成项（阶段三）

1. TOML 到 SimConfig 映射已覆盖 `simulation/neuron/network/evolution/evolution.fitness_weights`
2. CLI 覆盖优先级（命令行 > TOML > 默认）已测试保护
3. 默认 run 路径已通过“非零放电/传播”回归测试
4. 引擎统计可比性已建立测试基线
5. 文档 `schema_version` 与事件语义已统一

## 建议的下一阶段优先事项

1. **R8 AutoLearner 向量化集成**（补齐双引擎功能对等）
2. **R9 mypy 严格模式治理**（分模块推进）
3. **R1 Rust 运行时一致性治理**（环境脚本 + CI 导入校验）
4. **测试覆盖率提升**（visualization.py / core_bridge.py 盲区补齐）
