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

**调参建议**：
- 降低 `reproduction_threshold` → 更容易繁殖 → 种群扩张
- 增大 `energy_decay` → 能量流失快 → 适者生存压力增大
- `max_neurons` 决定种群上限，影响内存占用

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
| `energy_efficiency` | 0.5 | 能量效率权重（能量 / 放电数） |
| `prediction_hit` | 0.8 | 预测命中权重 |

**适应度公式**：
```
fitness = propagation_count * w_prop
        + (energy / max(1, spike_count)) * w_eff
        + prediction_hits * w_pred
```

繁殖时按适应度排序，高适应度优先繁殖。

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
