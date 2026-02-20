# 下一阶段风险清单（v0.3.x）

> 更新时间：2026-02-19
>
> 本清单基于阶段三当前实现状态：P0~P5 已闭环，输出治理能力（采样/压缩/轮转）已落地。类型注解 (mypy) 已通过检查。

---

## 🔴 高优先级风险（仍需处理）

### R1: 激活动力学对外部输入强依赖
**现象**: 在弱刺激配置下，放电/传播仍可能偏低。
**影响**: 实验有效窗口缩短，统计不稳定。
**建议**:
1. 引入 `background_current` 或 `spontaneous_spike_rate` 参数
2. 提供“低激活自动告警阈值”可配置化（而非固定提示）
3. 增加官方“激活态配置模板”

### R2: 能量经济仍可能出现双极化（爆发/衰亡）
**现象**: 某些配置在中后期出现快速扩张后回落。
**影响**: 长程实验可解释性下降。
**建议**:
1. 增加“生死平衡回归”长期指标（如 500 tick 生存区间）
2. 评估按比例衰减模型与线性衰减模型的可控性
3. 为常用实验提供建议参数区间

### R3: 向量化与原始引擎仍为”统计一致”而非”机制一致”
**现象**: 已建立统计可比测试，但模型细节尚未完全对齐。
**进展**: v0.3.1 已统一双引擎接口（`TickSnapshot` / `EngineSummary` / `SimEngine` 协议），返回字段完全一致。
**影响**: 机制级结论需谨慎外推。
**建议**:
1. 分阶段引入 Izh/AdEx 向量化实现
2. 在 benchmark 中持续标注”可比项/不可比项”
3. 增加更多趋势对照（PE、活跃率、能量分布）

---

## 🟡 中优先级风险

### R4: 配置参数存在“已解析未消费”字段
**现象**: 部分字段已进入 `SimConfig`，但核心引擎尚未完整消费（如预算相关参数）。
**建议**: 建立“已解析/已消费”矩阵并在版本发布前核对。

### R5: 学习引擎与仿真引擎耦合度低
**现象**: 规则发现需要手动 `discover(samples)`，未自动从网络提取。
**原因**: 学习引擎设计为独立模块。
**建议**:
1. 在 `Network.tick()` 中可选调用 learning callback
2. 添加 `AutoLearner` 包装类，自动从 neuron mem 提取样本

### R6: CLI 关键路径覆盖仍偏低
**现象**: 入口分发覆盖较完整，但 `cmd_run/cmd_benchmark` 冒烟路径覆盖不足。
**建议**:
1. 增加 `cmd_run` 最小运行回归（含输出文件存在性校验）
2. 增加 `cmd_benchmark` 冒烟测试（小规模参数）
3. 将 CLI 覆盖率纳入发布前检查清单

### R7: 长程输出治理（已闭环）
**状态**: 已完成 `log_interval + gzip + 轮转`。
**说明**:
1. 已支持 `--log-interval`
2. 已支持 `--gzip-output`
3. 已支持 `--max-file-size` 自动轮转
4. 已提供 `examples/decompress_outputs.py` 批量解压脚本

---

## 🟢 低优先级风险

### R8: 类型注解不完整 ✅ 已修复
**状态**: v0.3.1 已通过 `protocols.py` 定义统一 `TickSnapshot`、`EngineSummary` TypedDict 和 `SimEngine` Protocol，双引擎返回类型完全收敛。
**现象**: ~~部分方法返回 `dict` 而非具体 TypedDict。~~
**建议**: ~~定义 `SnapshotDict`, `MetricsDict` 等 TypedDict（可选）。~~ 已完成。

### R9: 无持久化/恢复功能
**建议**: 添加 checkpoint/restore — 将网络状态序列化为 JSON/pickle。

### R10: 空间系统未使用
**现象**: `SpatialGrid` 已实现但原始引擎中 `Neuron.pos` 未参与连接决策。
**建议**: 在 `Network` 中添加 `connect_by_proximity()` 方法。

### R11: 无可视化
**建议**: 添加 matplotlib / plotly 支持：
1. 种群曲线 (alive vs time)
2. PE 收敛曲线
3. 网络拓扑图

---

## 技术债务

| 编号 | 描述 | 影响 |
|------|------|------|
| TD-1 | `network.py` 中 `check_reproduction` 和 `check_death` 与 `tick` 紧耦合 | 难以单独测试 |
| TD-2 | `Event` 是普通 dataclass，无 `__lt__` — heapq 依赖 id 字段唯一性 | 可能在极端情况下报错 |
| TD-3 | `Neuron.observe()` 每次重新计算 transitions | O(n) 开销，高频调用时低效 |
| TD-4 | 日志依赖 `time.strftime` — 快速连续创建会冲突 | 建议加 ms 精度或序号 |

---

## 已完成项（阶段三）

1. TOML 到 SimConfig 映射已覆盖 `simulation/neuron/network/evolution/evolution.fitness_weights`
2. CLI 覆盖优先级（命令行 > TOML > 默认）已测试保护
3. 默认 run 路径已通过“非零放电/传播”回归测试
4. 引擎统计可比性已建立测试基线
5. 文档 `schema_version` 与事件语义已统一

## 建议的下一阶段优先事项

1. **R2 能量经济稳定性**（长程平衡策略）
2. **R3 深化引擎一致性**（机制级对照）
3. **R11 可视化**（标准曲线面板）
4. **R5 学习引擎自动接入**（从仿真流自动采样）
5. **R8 类型系统完善**（TypedDict 与接口收敛）
