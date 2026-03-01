# 可观测性指标（Metrics）

本文档基于以下代码事实：

- `src/neurogenesis_lab/protocols.py`
- `src/neurogenesis_lab/network.py`
- `src/neurogenesis_lab/vectorized.py`
- `src/neurogenesis_lab/observability.py`

## 1) TickSnapshot（引擎统一快照）

`tick()` 返回统一快照（`TickSnapshot`）。核心字段：

| 字段 | 说明 |
|------|------|
| `schema_version` | Schema 版本 |
| `tick` | 当前 tick |
| `alive_neurons` | 存活神经元数 |
| `total_neurons` | 总神经元数 |
| `tick_spikes` | 本 tick 放电数 |
| `propagation_count` | 本 tick 传播数 |
| `births` | 本 tick 出生数 |
| `deaths` | 本 tick 死亡数 |
| `prediction_error` | 本 tick 预测误差 |
| `mean_energy` | 存活神经元平均能量 |
| `ready_events` | 就绪事件数（两引擎均可提供） |
| `rule_count` | 可选，`AutoLearner` 注入 |
| `prediction_accuracy` | 可选，`AutoLearner` 注入 |

## 2) EngineSummary（引擎统一摘要）

`get_summary()` 返回统一摘要（`EngineSummary`）。

| 字段 | 说明 |
|------|------|
| `schema_version` | Schema 版本 |
| `total_neurons` | 总神经元数 |
| `alive_neurons` | 存活神经元数 |
| `total_spikes` | 累计放电数 |
| `total_propagations` | 累计传播数 |
| `birth_count` | 累计出生数 |
| `death_count` | 累计死亡数 |
| `pe_moving_avg` | PE 滑动均值 |
| `active_ratio` | 活跃占比 |
| `mean_energy` | 平均能量 |
| `corpse_pool_free` | 仅 `Network` |
| `corpse_pool_reused` | 仅 `Network` |

## 3) JSONL 报告结构（observer/logger）

`NetworkObserver.generate_report()` 生成结构：

```json
{
  "schema_version": "2.0.0",
  "metrics": {...},
  "learning": {...},
  "trends": {...},
  "debug": {...}
}
```

### 3.1 `metrics` 字段

来源：`NetworkObserver.get_network_metrics()`

- `total_neurons`
- `alive_neurons`
- `dead_neurons`
- `total_connections`
- `avg_connections`
- `mean_energy`
- `queue_size`
- `avg_novelty`
- `avg_emotion`
- `avg_entropy`
- `active_ratio`
- `corpse_count`

### 3.2 `learning` 字段

来源：`NetworkObserver.get_learning_metrics()`

- `total_spikes`
- `total_propagations`
- `pe_moving_avg`

### 3.3 `trends` 字段

来源：`NetworkObserver.get_trend_metrics()`

- `pe_moving_avg`
- `birth_rate`
- `death_rate`
- `alive_trend`
- `effective_propagation_rate`

## 4) 输出文件约定

`ObservabilityLogger` + `create_experiment_logger()`：

- 默认主文件：`run.jsonl`
- gzip：`run.jsonl.gz`（`--gzip-output`）
- 轮转：`run_0001.jsonl(.gz)`、`run_0002.jsonl(.gz)`（`--max-file-size > 0`）
- 采样：仅每 `N` tick 记录（`--log-interval N`）
- 异步写盘：`--async-log` + `--log-buffer-size`

说明：

- `config` 记录与 `summary` 记录不受 tick 采样跳过影响。
- 输出目录默认由 `run` 命令创建在 `outputs/` 下时间戳实验目录。

## 5) checkpoint 文件约定

`run` 命令按间隔输出：

- `checkpoint_tick_<global_tick>.pkl`

由 `--checkpoint-interval` 与 `--checkpoint-dir` 控制，`--resume-from` 恢复。

## 6) Rust 状态观测（info）

`python -m neurogenesis_lab info` 输出：

- `Rust Core: available|unavailable`
- 可用组件列表（来自 `core_bridge.rust_core_status()`）

用于确认 Rust 路径是否可用（特别是 benchmark 对照时）。
