# 参数说明

所有参数通过 `SimConfig` 集中管理，也可以通过 TOML 配置文件设置。

---

## `[simulation]` — 仿真全局参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ticks` | int | 100 | 仿真总步数 |
| `seed` | int | 42 | 随机种子（影响 random + numpy） |
| `dt` | float | 1.0 | 时间步长 |

---

## `[neuron]` — 神经元动力学参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `v_rest` | float | -65.0 | 静息电位 (mV) |
| `v_reset` | float | -65.0 | 重置电位 (mV) |
| `v_th` | float | -55.0 | 阈值电位 (mV)，低于此值不放电 |
| `tau_m` | float | 20.0 | 膜时间常数 (ms)，控制电位衰减速度 |
| `r_m` | float | 1.0 | 膜电阻 (MΩ)，影响输入电流对电位的影响 |
| `spike_cost` | float | 0.05 | 每次放电消耗的能量 |
| `input_gain` | float | 0.05 | 输入事件转换为能量的增益系数 |
| `mem_size` | int | 64 | 每个神经元的记忆队列容量 |

**调参建议**：
- 降低 `v_th` → 更容易放电 → 更活跃的网络
- 增大 `spike_cost` → 放电代价高 → 促进选择性放电
- 增大 `mem_size` → 更长的记忆 → 更多规则可发现

---

## `[network]` — 网络级参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reproduction_threshold` | float | 2.5 | 繁殖所需最低能量 |
| `reproduction_share` | float | 0.4 | 繁殖时子代获得的能量比例 |
| `mutation_scale` | float | 0.05 | 繁殖突变缩放因子 |
| `energy_decay` | float | 0.003 | 每 tick 能量自然衰减量 |
| `max_neurons` | int | 50 | 网络中同时存在的最大神经元数 |
| `max_links` | int | 200 | 网络中最大连接数 |
| `corpse_ttl` | int | 50 | 尸体存留 tick 数 |
| `scavenger_radius` | float | 2.5 | 拣食者范围 (距尸体的最大距离) |
| `input_current_scale` | float | 1.5 | 外部输入电流缩放 |
| `move_cost` | float | 0.01 | 神经元移动的能量消耗 |
| `energy_cap` | float | 5.0 | 神经元能量上限（防止无限积累） |
| `adaptive_decay` | bool | true | 是否按种群密度自适应调整能量衰减 |
| `adaptive_reproduction` | bool | true | 是否按种群规模自适应调整繁殖成本 |
| `endangered_threshold` | int | 10 | 濒危模式阈值（低于该值触发保护） |

**调参建议**：
- 降低 `reproduction_threshold` → 更容易繁殖 → 种群扩张
- 增大 `energy_decay` → 能量流失快 → 适者生存压力增大
- `max_neurons` 决定种群上限，影响内存占用
- 适当降低 `energy_cap` 可抑制个体能量垄断
- `adaptive_decay/adaptive_reproduction` 适合长程实验，建议默认开启

---

## `[evolution]` — 演化系统参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mutation_amplitude` | float | 0.05 | 突变振幅 |
| `energy_budget_per_tick` | float | 0.0 | 每 tick 能量预算（0 = 不限） |
| `congestion_penalty` | float | 0.1 | 拥堵惩罚系数（连接数 > 10 时生效） |
| `min_neurons` | int | 2 | 最小存活神经元数（保底复活机制） |

### `[evolution.fitness_weights]` — 适应度权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `propagation` | 1.0 | 传播次数权重 |
| `energy_efficiency` | 0.5 | 当前能量权重（能量越高得分越高） |
| `prediction_hit` | 0.8 | 预测命中权重 |

**适应度公式**：
```
fitness = propagation_count * w_prop
        + energy * w_eff
        + prediction_hits * w_pred
```

繁殖时按适应度排序，高适应度优先繁殖。

> 注：`mutation_amplitude` 与 `energy_budget_per_tick` 已进入配置与日志输出，
> 在当前版本中已直接接入运行时逻辑：
> - `mutation_amplitude`：用于繁殖阶段的突变幅度（覆盖 network.mutation_scale）。
> - `energy_budget_per_tick`：每 tick 按存活神经元均分注入能量预算。

### 参数消费矩阵（解析 → 运行时）

| 参数 | 解析入口 | 运行时消费点 | 当前状态 |
|------|----------|--------------|----------|
| `simulation.ticks` | `_build_sim_config()` | `cmd_run()` 主循环边界 | ✅ 已消费 |
| `simulation.seed` | `_build_sim_config()` | `seed_all()` + 引擎初始化 | ✅ 已消费 |
| `simulation.dt` | `_build_sim_config()` | `network.tick(t, dt=cfg.dt)` | ✅ 已消费 |
| `network.mutation_scale` | `_build_sim_config()` | `Network.check_reproduction()` 后备突变幅度 | ✅ 已消费 |
| `evolution.mutation_amplitude` | `_build_sim_config()` | `Network.check_reproduction()` 优先突变幅度 | ✅ 已消费 |
| `evolution.energy_budget_per_tick` | `_build_sim_config()` | `Network.tick()` 每 tick 能量预算注入 | ✅ 已消费 |
| `evolution.congestion_penalty` | `_build_sim_config()` | `Network.check_reproduction()` 拥塞惩罚 | ✅ 已消费 |
| `evolution.min_neurons` | `_build_sim_config()` | `Network.check_death()` 保底机制 | ✅ 已消费 |

**调参建议**：
- 增大 `prediction_hit` → 奖励预测准确的神经元
- 增大 `congestion_penalty` → 抑制过度连接
- `min_neurons` 防止种群灭绝

---

## 配置文件示例

```toml
[simulation]
ticks = 200
seed = 42

[neuron]
v_th = -48.0
spike_cost = 0.04

[network]
max_neurons = 80
energy_decay = 0.003

[evolution]
min_neurons = 5
congestion_penalty = 0.15

[evolution.fitness_weights]
propagation = 1.0
energy_efficiency = 0.8
prediction_hit = 1.0
```

## 命令行覆盖

命令行参数优先级高于 TOML 配置文件：

```bash
python -m neurogenesis_lab run --config config.toml --ticks 500 --seed 99
```

上例中 `ticks=500` 和 `seed=99` 会覆盖 TOML 中的值。

### 常用 `run` CLI 参数（补充）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--neurons` | int | 8 | 初始神经元数量 |
| `--expr` | str | `1+2=3` | 外部刺激表达式 |
| `--input-weight` | float | 40.0 | 外部注入事件默认权重 |
| `--stim-interval` | int | 5 | 外部刺激注入间隔（tick） |
| `--output` | str | `outputs` | 输出目录 |
| `--config` | str | None | TOML 配置路径 |
| `--debug` | flag | 关闭 | 输出调试级追踪信息 |

---

## 输出治理参数（CLI）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--log-interval` | int | 1 | 日志采样间隔（每 N tick 记录一次） |
| `--gzip-output` | flag | 关闭 | 输出为 `.jsonl.gz` 压缩文件 |
| `--max-file-size` | int | 0 | 单文件最大字节数，`>0` 时启用自动轮转 |
| `--async-log` | flag | 关闭 | 启用异步双缓冲日志写入 |
| `--log-buffer-size` | int | 256 | 异步双缓冲阈值（记录条数） |

示例：

```bash
python -m neurogenesis_lab run --ticks 500 --log-interval 10 --gzip-output --max-file-size 5000000
```

该命令会每 10 tick 采样一次日志，输出为压缩 JSONL，并在单文件超过 5MB 时自动轮转为分段文件。

高频记录建议：

```bash
python -m neurogenesis_lab run --ticks 500 --log-interval 1 --async-log --log-buffer-size 512
```

该配置可降低主循环被 I/O 阻塞的概率。

---

## 可视化参数（CLI）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--visualize` | flag | 关闭 | 启动实时可视化监控面板 |
| `--plot-interval` | int | 5 | 图表更新间隔（每 N tick） |

示例：

```bash
python -m neurogenesis_lab run --visualize --plot-interval 10
```

---

## 基准参数（CLI）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--neurons` | int | 1000 | 基准主规模（用于原始/向量化对照） |
| `--ticks` | int | 100 | 基准步数 |
| `--seed` | int | 42 | 随机种子 |
| `--rust-size` | int | 10000 | Rust 并行 LIF 微基准神经元数量 |

示例：

```bash
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100 --rust-size 20000
```

---

## 持久化与恢复参数（CLI）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint-interval` | int | 0 | 每 N 个本次运行 tick 保存一次 checkpoint，0=禁用 |
| `--checkpoint-dir` | str | `outputs` | checkpoint 输出目录 |
| `--resume-from` | str | None | 从指定 checkpoint 文件恢复运行 |

示例：

```bash
# 自动保存 checkpoint
python -m neurogenesis_lab run --ticks 500 --checkpoint-interval 100 --checkpoint-dir outputs/checkpoints

# 从 checkpoint 恢复并继续运行
python -m neurogenesis_lab run --resume-from outputs/checkpoints/checkpoint_tick_500.pkl --ticks 200
```
