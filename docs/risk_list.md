# 下一阶段风险清单

> v0.2.0 → v0.3.0 迭代中需要关注的技术风险与潜在问题

---

## 🔴 高优先级风险

### R1: 神经元放电率过低
**现象**: 标准实验中 `total_spikes = 0`，神经元未达到阈值电位。
**原因**: 输入信号通过 `input_gain=0.05` 转换后能量增加微弱，而 LIF/Izh/AdEx 模型的 `v_th` 默认为 -50.0，需要较强的持续输入才能触发。
**影响**: 放电率为零 → 无传播 → 无繁殖 → 演化系统闲置。
**建议**:
1. 调低默认 `v_th`（如 -55.0）或增大 `input_gain`（如 0.2）
2. 添加背景噪声电流 (background current)
3. 在 `receive()` 中累积能量直接影响膜电位
4. 考虑添加 `spontaneous_spike_rate` 参数

### R2: 能量衰减导致种群快速灭绝
**现象**: 200 ticks 后 15/15 神经元死亡，仅靠 `min_neurons` 保底。
**原因**: `energy_decay=0.005` 每 tick 线性扣减，而无有效能量补充来源（放电率为零）。
**影响**: 演化实验的有效窗口过短。
**建议**:
1. 引入能量补充机制（如 scavenger 成功时补充更多）
2. 考虑 `energy_reward_on_receive` 参数
3. 将衰减改为按比例衰减: `energy *= (1 - decay_rate)` 而非线性扣减

### R3: 向量化引擎与原始引擎行为不一致
**现象**: `VectorizedNetwork` 使用纯 LIF 模型，不支持 Izh/AdEx。
**原因**: NumPy 向量化要求统一更新方程。
**影响**: 大规模实验结果与小规模原始引擎不可直接对比。
**建议**:
1. 在 VectorizedNetwork 中添加 model_type 字段
2. 实现向量化版本的 Izh 更新 (两个数组: v, u)
3. 添加对比验证测试 (相同参数下两引擎行为应一致)

---

## 🟡 中优先级风险

### R4: TOML 配置仅部分加载
**现象**: `__main__.py` 的 `_build_sim_config()` 只读取 `[simulation]` 中的 `ticks` 和 `seed`。
**原因**: 初版 CLI 简化实现，未映射 `[neuron]`、`[network]`、`[evolution]` 到 SimConfig 子结构。
**建议**: 添加完整的 TOML → SimConfig 映射，包括嵌套参数。

### R5: 学习引擎与仿真引擎耦合度低
**现象**: 规则发现需要手动 `discover(samples)`，未自动从网络提取。
**原因**: 学习引擎设计为独立模块。
**建议**:
1. 在 `Network.tick()` 中可选调用 learning callback
2. 添加 `AutoLearner` 包装类，自动从 neuron mem 提取样本

### R6: 无测试覆盖率报告
**建议**: 添加 `pytest-cov`，目标覆盖率 > 80%。

### R7: 输出文件增长不可控
**现象**: 长程实验（500+ ticks）的 JSONL 可能很大。
**建议**:
1. 添加 `log_interval` — 每 N tick 记录一次
2. 支持压缩输出 (.jsonl.gz)
3. 添加 `max_file_size` 自动轮转

---

## 🟢 低优先级风险

### R8: 类型注解不完整
**现象**: 部分方法返回 `dict` 而非具体 TypedDict。
**建议**: 定义 `SnapshotDict`, `MetricsDict` 等 TypedDict。

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

## 建议的下一阶段优先事项

1. **解决 R1 (放电率为零)** — 这是最关键的功能缺陷
2. **解决 R2 (能量经济平衡)** — 与 R1 联动
3. **完善 R4 (TOML 全映射)** — 提升用户体验
4. **添加 R11 (可视化)** — 提高实验可解读性
5. **解决 R3 (引擎一致性)** — 为大规模实验做准备
