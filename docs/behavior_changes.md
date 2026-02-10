# 行为变更文档 — v0.1.0 → v0.2.0

> 本文档记录 v0.2.0 相对于 v0.1.0 (开发清单01基线) 的所有行为变更。

---

## 1. 配置系统

| 变更 | 旧行为 (v0.1.0) | 新行为 (v0.2.0) |
|------|-----------------|-----------------|
| 配置结构 | 单层 `Config(ticks, network_size, energy_decay, seed)` | 分层 `SimConfig(ticks, seed, dt, neuron, network, evolution)` |
| 随机种子 | `random.seed()` 分散在各处 | `seed_all()` 统一入口，覆盖 random + numpy |
| 参数验证 | 无 | 每层 `validate()` 方法返回错误列表 |
| Schema 版本 | 无 | `SCHEMA_VERSION = "2.0.0"` 嵌入所有输出 |
| 向后兼容 | - | `Config` 类保留，可继续使用 |

**影响**: 旧的 `Config()` 仍可用但功能有限。推荐迁移到 `SimConfig`。

---

## 2. 神经元模型

| 变更 | 旧行为 | 新行为 |
|------|--------|--------|
| NaN/Inf 防护 | 无 — 可能产生 NaN 传播 | `safe_float()` 包裹所有输入/输出，NaN/Inf 自动回退到 0.0 |
| AdEx 溢出 | `exp()` 可能溢出 → crash | exp_arg 钳制到 20.0 (≈ 4.8e8)，永不溢出 |
| 适应度追踪 | 无 | 新增 `spike_count`, `propagation_count`, `prediction_hits` |
| `fitness()` 方法 | 无 | 加权计算: propagation × w₁ + efficiency × w₂ + prediction × w₃ |
| `move_cost` | 硬编码 0.01 | 可配置，通过 `SimConfig.network.move_cost` |
| `emit()` | 可能发出负 delay | 负 delay 钳制为 0 |
| `spawn()` | 突变无边界保护 | 突变结果通过 `safe_float()` 过滤 |

**影响**: 数值稳定性大幅提升。之前可能因 NaN 导致的静默错误现已消除。

---

## 3. 网络引擎

| 变更 | 旧行为 | 新行为 |
|------|--------|--------|
| `tick()` 返回值 | 简单 snapshot | 新增 `schema_version`, `tick_spikes`, `births`, `deaths`, `total_neurons` |
| 繁殖策略 | 随机选择 | **适应度排序** — 高 fitness 优先繁殖 |
| 拥堵控制 | 无 | `congestion_penalty` — 连接数 > 10 时提高繁殖门槛 |
| 种群保底 | 无 — 可全灭 | `min_neurons=2` — 存活数不足时自动复活最高能量尸体 |
| 尸体系统 | 固定能量 | 新增 `decay_rate`，尸体能量逐 tick 衰减 |
| PE 历史 | 无 | `_pe_history` 列表 + `pe_moving_average(window=10)` |
| 统计追踪 | 无 | `birth_count`, `death_count`, `total_spikes`, `total_propagations` |
| `get_summary()` | 无 | 完整网络状态概要 |
| `active_neuron_ratio()` | 无 | 存活比例实时计算 |

**破坏性变更**: `check_death()` 不再允许种群灭绝（除非显式设置 `min_neurons=0`）。
旧测试 `test_death_removes_neuron_and_creates_corpse` 需添加 `min_neurons=0`。

---

## 4. 学习引擎

| 变更 | 旧行为 | 新行为 |
|------|--------|--------|
| PatternMiner | 只统计频次 | 新增 `support_count` + `confidence()` 方法 |
| QueryEngine | 返回纯答案 | 新增 `query_with_explanation()` 返回 `{answer, reason, confidence, rules_hit}` |
| RuleStore | 只有 trigram/bigram lookup | 新增 `trigram_confidence()`, `bigram_confidence()`, `top_candidates_middle()`, `top_candidates_next()` |
| RuleComposer | 只支持加法 | 新增 `_apply_simple_subtraction()`, `_apply_simple_multiply()` |
| RuleComposer | 无解释 | 新增 `apply_with_explanation()` 返回 `{result, reason, rules_hit}` |
| 冲突策略 | 无 | `conflict_strategy` 字段 ("highest_support" / "weighted_score") |

**影响**: 所有旧 API 保持兼容。新方法为纯增量添加。

---

## 5. 可观察性

| 变更 | 旧行为 | 新行为 |
|------|--------|--------|
| 指标层级 | 单层 tick_metrics | 三层: `network_metrics`, `neuron_metrics`, `learning_metrics` |
| 趋势追踪 | 无 | `get_trend_metrics()`: PE 移动平均、出生率、死亡率、有效传播率 |
| 神经元过滤 | 无 | `neuron_filter` 回调 — 条件性追踪 (如 energy > 3.0) |
| 实验管理 | 单文件覆盖 | `create_experiment_logger()` — 自动时间戳分目录 |
| 配置记录 | 无 | `log_config()` — 实验参数写入 JSONL 开头 |
| Schema 标记 | 无 | 所有输出记录包含 `schema_version` |

**影响**: 旧的 `get_tick_metrics()` / `get_neuron_detail()` 保留为兼容别名。

---

## 6. 新增功能

### 向量化引擎 (`vectorized.py`)
- `VectorizedNetwork`: NumPy 数组存储所有神经元状态，支持万级神经元
- `SpatialGrid`: 网格空间索引，`query_radius()` 快速邻域查询
- 方法: `init_neurons()`, `connect_radius()`, `connect_knn()`, `tick()`, `get_summary()`
- 繁殖利用 dead slots，无需扩展数组
- NaN/Inf 通过 `np.isfinite()` 掩码保护

### CLI 入口 (`__main__.py`)
- `python -m neurogenesis_lab run` — 运行仿真
- `python -m neurogenesis_lab benchmark` — 性能测试
- `python -m neurogenesis_lab info` — 版本信息
- 支持 `--config TOML`、`--seed`、`--ticks` 覆盖

### TOML 配置
- `config.toml` — 默认配置模板
- `experiments/` — 三个实验配置模板

---

## 7. 测试覆盖变更

| 指标 | v0.1.0 | v0.2.0 |
|------|--------|--------|
| 测试文件数 | 6 | 13 |
| 测试用例数 | 7 | 153+ |
| 覆盖模块 | 集成测试为主 | config, encoding, neuron, network, learning, observability, vectorized |
| 行为测试 | 无 | spike 传播、PE 收敛、生死平衡、NaN 防护 |

---

## 8. 依赖变更

| 依赖 | v0.1.0 | v0.2.0 |
|------|--------|--------|
| numpy | 无 | `>=1.24` (向量化引擎所需) |
| pytest | 开发依赖 | 开发依赖 (不变) |

---

## 迁移指南

### 最小迁移（保持旧代码运行）
无需改动。`Config` 类、`get_tick_metrics()`、`get_neuron_detail()` 均保留。

### 推荐迁移
```python
# 旧
from neurogenesis_lab.config import Config
cfg = Config(ticks=100, seed=42)

# 新
from neurogenesis_lab.config import SimConfig, seed_all
cfg = SimConfig(ticks=100, seed=42)
seed_all(cfg.seed)
errors = cfg.validate()
```
