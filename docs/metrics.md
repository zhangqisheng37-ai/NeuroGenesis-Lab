# 可观测性指标 (Observability Metrics)

NeuroGenesis Lab 提供了全链路的数据观测能力。所有实验数据以 JSONL (JSON Lines) 格式流式写入磁盘，支持实时分析与离线回放。

## 数据结构

每行日志记录代表一个 Tick 的完整状态快照，包含以下核心字段：

```json
{
  "schema_version": "2.0.0",
  "type": "report",
  "tick": 123,
  "metrics": { ... },    // 网络级指标
  "learning": { ... },   // 学习与预测指标
  "trends": { ... },     // 趋势与统计
  "debug": { ... }       // 详细追踪（可选）
}
```

## 1. 网络指标 (Network Metrics)

位于 `metrics` 字段，反映种群的宏观状态：

| 字段名 | 说明 |
|--------|------|
| `total_neurons` | 总神经元数（含刚死亡未清理的） |
| `alive_neurons` | 存活神经元数 |
| `dead_neurons` | 死亡神经元数（不含已清理尸体） |
| `total_connections` | 全网突触连接总数 |
| `avg_connections` | 平均每个神经元的连接数 |
| `avg_energy` | 存活神经元的平均能量 |
| `avg_novelty` | 平均新颖性 (0.0~1.0)，反映输入的多样性 |
| `avg_entropy` | 平均信息熵，反映状态复杂度 |
| `avg_emotion` | 平均情感值 (保留字段) |
| `active_ratio` | 活跃比例 (Alive / Total) |
| `queue_size` | 事件队列中的待处理事件数 |
| `corpse_count` | 场景中存在的尸体数量 |

## 2. 学习指标 (Learning Metrics)

位于 `learning` 字段，反映系统的认知与计算能力：

| 字段名 | 说明 |
|--------|------|
| `total_spikes` | 累计放电次数 |
| `total_propagations` | 累计成功传播的事件数 |
| `pe_moving_avg` | 预测误差 (Prediction Error) 的滑动均值 |

## 3. 趋势指标 (Trend Metrics)

位于 `trends` 字段，用于分析演化动力学：

| 字段名 | 说明 |
|--------|------|
| `birth_rate` | 最近 N 个 tick 的平均出生率 |
| `death_rate` | 最近 N 个 tick 的平均死亡率 |
| `effective_propagation_rate` | 有效传播率 (Total Propagations / Total Spikes) |
| `alive_trend` | 最近 N 个 tick 的存活数序列 |

## 4. 神经元详情 (Neuron Detail)

当开启 Debug 模式或指定追踪 ID 时，`debug.tracked_neurons` 列表会包含详细的神经元状态：

| 字段名 | 说明 |
|--------|------|
| `neuron_id` | 唯一标识符 (UUID) |
| `pos` | 坐标 (x, y) |
| `energy` | 当前能量 |
| `v` | 当前膜电位 |
| `spike_count` | 累计放电次数 |
| `recent_tokens` | 短期记忆中的 Token 序列 |
| `top_transitions` | 观测到的高频 Token 转移 (Pattern) |
| `states` | 内部状态字典 (novelty, entropy 等) |

## 输出文件

默认情况下，实验数据输出至 `outputs/<experiment_id>/run.jsonl`。

可以使用 `examples/observe_jsonl.py` 或自定义脚本对 JSONL 文件进行解析与可视化。
