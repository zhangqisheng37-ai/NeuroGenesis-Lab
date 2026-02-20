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

## 0. Tick 快照 (TickSnapshot)

`tick()` 方法返回的统一快照类型（`Network` 和 `VectorizedNetwork` 共用），定义于 `protocols.py`：

| 字段名 | 说明 |
|--------|------|
| `schema_version` | 数据格式版本 |
| `tick` | 当前 tick 编号 |
| `alive_neurons` | 存活神经元数 |
| `total_neurons` | 总神经元数 |
| `tick_spikes` | 本 tick 放电次数 |
| `propagation_count` | 本 tick 传播事件数 |
| `births` | 本 tick 出生数 |
| `deaths` | 本 tick 死亡数 |
| `prediction_error` | 本 tick 预测误差 |
| `mean_energy` | 存活神经元平均能量 |

> 注：`Network` 引擎额外返回 `ready_events` 字段（就绪事件数），不在统一协议中但可直接访问。

## 0.1 引擎摘要 (EngineSummary)

`get_summary()` 方法返回的统一摘要类型：

| 字段名 | 说明 |
|--------|------|
| `schema_version` | 数据格式版本 |
| `total_neurons` | 总神经元数 |
| `alive_neurons` | 存活神经元数 |
| `total_spikes` | 累计放电次数 |
| `total_propagations` | 累计传播事件数 |
| `birth_count` | 累计出生数 |
| `death_count` | 累计死亡数 |
| `pe_moving_avg` | 预测误差滑动均值 |
| `active_ratio` | 活跃比例 (Alive / Total) |
| `mean_energy` | 存活神经元平均能量 |

## 1. 网络指标 (Network Metrics)

位于 `metrics` 字段，反映种群的宏观状态：

| 字段名 | 说明 |
|--------|------|
| `total_neurons` | 总神经元数（含刚死亡未清理的） |
| `alive_neurons` | 存活神经元数 |
| `dead_neurons` | 死亡神经元数（不含已清理尸体） |
| `total_connections` | 全网突触连接总数 |
| `avg_connections` | 平均每个神经元的连接数 |
| `mean_energy` | 存活神经元的平均能量 |
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

当启用输出治理参数时：

- `--gzip-output`：输出为 `run.jsonl.gz`
- `--max-file-size N`（N>0）：超过阈值后自动轮转为 `run_0001.jsonl(.gz)`, `run_0002.jsonl(.gz)` ...
- `--log-interval N`：每 N tick 记录一次 tick 报告（`config` 与 `summary` 仍会记录）

可以使用 `examples/observe_jsonl.py` 或自定义脚本对 JSONL 文件进行解析与可视化。

若文件为 `.jsonl.gz`，可使用：

```bash
python examples/decompress_outputs.py --path outputs --recursive
```

将压缩结果批量解压为同目录下的 `.jsonl` 文本文件。

## 事件字段语义（Event）

编码器和网络内部事件都使用 `Event(token_id, t, src_id, meta)` 结构。

| 字段名 | 语义 |
|--------|------|
| `token_id` | 当前脉冲携带的符号 ID |
| `t` | 事件计划投递的 tick |
| `src_id` | **本次事件要投递到的目标神经元 ID** |
| `meta.from` | 传播来源神经元 ID（仅网络传播事件有） |
| `meta.weight` | 突触权重或外部注入权重 |

> 注意：当前实现中 `src_id` 是“目标路由 ID”，不是“发送方 ID”。发送方通过 `meta.from` 记录。

### 时序示例（delay=1）

1. tick=5：神经元 A 放电，向 B 发出传播事件（`src_id=B`, `meta.from=A`, `t=6`）
2. tick=6：网络队列投递该事件到 B
3. tick=6：B 的 `receive()` 被触发，进入后续膜电位更新与可能放电
