# 参数说明

本文档以以下代码为唯一事实来源：

- CLI 参数：`src/neurogenesis_lab/cli/parser.py`
- CLI→配置构建：`src/neurogenesis_lab/cli/config_builder.py`
- 默认值与校验：`src/neurogenesis_lab/config.py`
- 运行时消费：`src/neurogenesis_lab/cli/run.py`、`src/neurogenesis_lab/network.py`

---

## 1. 配置层参数（`SimConfig`）

### 1.1 `[simulation]`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ticks` | int | `100` | 本次运行 tick 数 |
| `seed` | int | `42` | 随机种子（`random` + `numpy`） |
| `dt` | float | `1.0` | 时间步长 |

### 1.2 `[neuron]`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `v_rest` | float | `-65.0` | 静息电位 |
| `v_reset` | float | `-65.0` | 放电重置电位 |
| `v_th` | float | `-55.0` | 放电阈值 |
| `tau_m` | float | `20.0` | 膜时间常数 |
| `r_m` | float | `1.0` | 膜电阻缩放 |
| `spike_cost` | float | `0.05` | 每次放电能量消耗 |
| `input_gain` | float | `0.05` | 输入转能量增益 |
| `mem_size` | int | `64` | 记忆窗口大小 |

### 1.3 `[network]`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reproduction_threshold` | float | `2.5` | 繁殖阈值 |
| `reproduction_share` | float | `0.4` | 繁殖时子代能量占比 |
| `mutation_scale` | float | `0.05` | 网络层突变幅度（后备） |
| `energy_decay` | float | `0.003` | 每 tick 基础衰减 |
| `max_neurons` | int | `50` | 最大神经元数 |
| `max_links` | int | `200` | 最大连接数 |
| `corpse_ttl` | int | `50` | 尸体生存 tick |
| `scavenger_radius` | float | `2.5` | 拾荒半径 |
| `input_current_scale` | float | `1.5` | 输入电流缩放 |
| `background_current` | float | `0.0` | 每 tick 背景电流 |
| `spontaneous_spike_rate` | float | `0.0` | 自发放电概率 `[0,1]` |
| `low_activation_threshold` | int | `0` | 低激活告警阈值（`0` 关闭） |
| `move_cost` | float | `0.01` | 移动成本 |
| `energy_cap` | float | `5.0` | 能量上限 |
| `adaptive_decay` | bool | `true` | 自适应衰减 |
| `adaptive_reproduction` | bool | `true` | 自适应繁殖成本 |
| `endangered_threshold` | int | `10` | 濒危阈值 |

### 1.4 `[evolution]`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mutation_amplitude` | float | `0.05` | 演化阶段优先突变幅度 |
| `energy_budget_per_tick` | float | `0.0` | 每 tick 全局能量预算 |
| `congestion_penalty` | float | `0.1` | 拥塞惩罚系数 |
| `min_neurons` | int | `2` | 最小存活数保护阈值 |

#### `[evolution.fitness_weights]`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `propagation` | `1.0` | 传播权重 |
| `energy_efficiency` | `0.5` | 能量效率权重 |
| `prediction_hit` | `0.8` | 预测命中权重 |

---

## 2. CLI 参数（`run`）

来源：`src/neurogenesis_lab/cli/parser.py`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--ticks` | int | `None` | 覆盖配置 `simulation.ticks` |
| `--seed` | int | `None` | 覆盖配置 `simulation.seed` |
| `--dt` | float | `None` | 覆盖配置 `simulation.dt` |
| `--neurons` | int | `8` | 初始神经元数 |
| `--expr` | str | `1+2=3` | 输入表达式 |
| `--input-weight` | float | `40.0` | 外部注入事件权重 |
| `--stim-interval` | int | `5` | 外部刺激注入间隔 |
| `--background-current` | float | `None` | 覆盖 `network.background_current` |
| `--spontaneous-spike-rate` | float | `None` | 覆盖 `network.spontaneous_spike_rate` |
| `--low-activation-threshold` | int | `None` | 覆盖 `network.low_activation_threshold` |
| `--output` | str | `outputs` | 输出目录 |
| `--config` | str | `None` | TOML 路径 |
| `--debug` | flag | `False` | debug 记录 |
| `--log-interval` | int | `1` | 每 N tick 记录一次 |
| `--gzip-output` | flag | `False` | 输出 `.jsonl.gz` |
| `--max-file-size` | int | `0` | >0 开启轮转 |
| `--async-log` | flag | `False` | 异步双缓冲写盘 |
| `--log-buffer-size` | int | `256` | 异步缓冲阈值 |
| `--visualize` | flag | `False` | 启动实时可视化 |
| `--plot-interval` | int | `5` | 可视化更新间隔 |
| `--checkpoint-interval` | int | `0` | 每 N 个本次运行 tick 保存 checkpoint |
| `--resume-from` | str | `None` | 从 checkpoint 恢复 |
| `--checkpoint-dir` | str | `outputs` | checkpoint 输出目录 |

---

## 3. CLI 参数（`benchmark`）

来源：`src/neurogenesis_lab/cli/parser.py`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--neurons` | int | `1000` | 基准神经元规模 |
| `--ticks` | int | `100` | 基准 tick 数 |
| `--seed` | int | `42` | 随机种子 |
| `--rust-size` | int | `10000` | Rust 并行 LIF 微基准规模 |

---

## 4. 参数优先级（重要）

来源：`build_sim_config()`（`src/neurogenesis_lab/cli/config_builder.py`）

优先级：

1. CLI 显式传入值（最高）
2. TOML 配置值（`--config`）
3. `SimConfig` / 子配置 dataclass 默认值（最低）

示例：

```bash
python -m neurogenesis_lab run --config experiments/experiment_stable.toml --ticks 500 --seed 99
```

上例中，`ticks` 和 `seed` 以 CLI 为准。

---

## 5. 运行时消费映射（关键字段）

| 字段 | 消费位置 | 行为 |
|------|----------|------|
| `simulation.ticks` | `cli/run.py:_init_network()` + 主循环 | 决定运行边界 |
| `simulation.seed` | `cli/run.py:cmd_run()` | `seed_all()` |
| `simulation.dt` | `cli/run.py:_run_simulation_loop()` | 传入 `network.tick(t, dt=...)` |
| `network.*` | `cli/run.py:_init_network()` | 构造 `Network` 参数 |
| `evolution.mutation_amplitude` | `Network(..., evolution_mutation_amplitude=...)` | 繁殖突变幅度优先值 |
| `evolution.energy_budget_per_tick` | `Network(..., energy_budget_per_tick=...)` | 每 tick 预算注入 |
| `--log-interval` / `--gzip-output` / `--max-file-size` / `--async-log` / `--log-buffer-size` | `cli/run.py:_init_logging()` → `create_experiment_logger()` | 输出治理与写盘模式 |
| `--checkpoint-interval` / `--checkpoint-dir` | `cli/run.py:_run_simulation_loop()` | 定期保存 checkpoint |
| `--resume-from` | `cli/run.py:_init_network()` | `Network.load_checkpoint()` 恢复 |

---

## 6. 版本与 Schema

- 代码版本：`0.6.0`（`src/neurogenesis_lab/__init__.py`）
- 数据 Schema：`2.0.0`（`src/neurogenesis_lab/config.py`）

可通过以下命令查看当前运行时口径：

```bash
python -m neurogenesis_lab info
```
