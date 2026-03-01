# 案例研究：规则涌现与学习引擎

NeuroGenesis Lab 的学习引擎（`AutoLearner`）通过观察神经元的记忆序列，自动发现符号规则并进行预测。本案例研究将探讨在长程仿真中观察到的规则涌现现象。

## 1. 实验设置

我们使用 `experiment_learning.toml` 配置运行了一个 300 ticks 的实验。该配置通过提高 `prediction_hit` 适应度权重，增强了对预测能力的选择压力，从而更容易观察到规则涌现。

```bash
python -m neurogenesis_lab run --config experiments/experiment_learning.toml --visualize --async-log --log-buffer-size 512
```

## 2. 观察到的现象

### 2.1 规则数量的增长

在实验初期，神经元的记忆序列较短，发现的规则数量较少。随着时间的推移，神经元积累了更多的记忆，`AutoLearner` 开始发现更多的 N-gram 规则。

- **初期**：规则数量缓慢增长，主要是一些简单的 2-gram 规则（如 `A -> B`）。
- **中期**：规则数量快速增长，出现了更复杂的 3-gram 和 4-gram 规则。
- **后期**：由于规则库容量上限（`max_rules`）的限制，规则数量趋于稳定。LRU 淘汰机制开始淘汰低支持度的规则，保留高支持度的规则。

### 2.2 预测准确率的提升

随着规则数量的增加，神经元的预测准确率也随之提升。

- **初期**：预测准确率较低，通常在 0.1-0.2 之间。
- **中期**：预测准确率稳步上升，达到 0.4-0.6。
- **后期**：预测准确率趋于稳定，通常在 0.7-0.9 之间。

### 2.3 预测驱动的能量分配

在启用了预测奖励的实验中，预测准确率高的神经元能够获得更多的能量奖励。这导致了能量分布的进一步分化。

- **高预测准确率神经元**：能量较高，更容易繁殖，将优秀的规则传递给子代。
- **低预测准确率神经元**：能量较低，容易死亡，被淘汰。

这种机制促进了整个种群预测能力的提升，形成了一种"适者生存"的演化动力学。

## 3. 结论

NeuroGenesis Lab 的学习引擎能够有效地从神经元的记忆序列中发现符号规则，并通过预测奖励机制促进种群预测能力的提升。这种结合了符号学习和神经演化的混合架构，为研究人工生命的认知能力提供了新的视角。

## 4. 附录：自适应稳定机制消融对照（2026-02-22）

为量化 `adaptive_decay` 与 `adaptive_reproduction` 的收益，补充了一组同种子消融实验。

### 4.1 实验配置与运行方式

- 同一随机种子：`seed=42`
- 同一仿真长度：`ticks=500`
- 唯一变量：`adaptive_decay` / `adaptive_reproduction` 开关

```bash
# 开启组
python -m neurogenesis_lab run --config experiments/ablation_adaptive_on.toml --output outputs/ablation_on

# 关闭组
python -m neurogenesis_lab run --config experiments/ablation_adaptive_off.toml --output outputs/ablation_off
```

实验输出：

- 开启组：`outputs/ablation_on/run_20260222_121401_330/run.jsonl`
- 关闭组：`outputs/ablation_off/run_20260222_121425_847/run.jsonl`

### 4.2 量化结果（ON vs OFF）

| 指标 | 开启 (ON) | 关闭 (OFF) | 变化（ON-OFF） |
|------|-----------|------------|----------------|
| `alive_mean` | 54.840 | 54.464 | +0.376 |
| `alive_std` | 13.628 | 14.059 | -0.431 |
| `pe_mean` | 0.02992 | 0.03104 | -0.00112 |
| `birth_rate_mean` | 0.3360 | 0.3578 | -0.0218 |
| `death_rate_mean` | 0.2322 | 0.2550 | -0.0228 |

运行摘要（终端汇总）：

- ON：`total_spikes=101`，`total_propagations=2124`，`births=168`，`deaths=117`，`prediction_accuracy=0.955`
- OFF：`total_spikes=100`，`total_propagations=2200`，`births=181`，`deaths=130`，`prediction_accuracy=0.957`

### 4.3 结论

该消融结果表明，自适应机制的主要收益体现在**稳定性增强**而非学习上限提升：

1. 种群波动降低（`alive_std` 下降）。
2. 预测误差均值降低（`pe_mean` 下降）。
3. 生死周转率降低（`birth_rate_mean`、`death_rate_mean` 同时下降）。

在本组参数下，学习准确率两组接近（OFF 略高 `0.002`），说明该机制更偏向生态稳态调节，而非直接抬升预测准确率上限。
