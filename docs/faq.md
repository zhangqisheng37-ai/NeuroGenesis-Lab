# FAQ — 常见问题

## 1. 神经元全部死亡了怎么办？

v0.2.0 引入了 `min_neurons` 保底机制（默认 = 2）。当存活数低于 `min_neurons` 时，系统会自动复活能量最高的尸体（energy 重置为 0.1）。

如需完全关闭保底：
```python
from neurogenesis_lab.network import Network
net = Network(neurons=..., min_neurons=0)
```

或在 TOML 中：
```toml
[evolution]
min_neurons = 0
```

## 2. 仿真速度太慢？

对于 1000+ 神经元的大规模仿真，请使用向量化引擎：

```python
from neurogenesis_lab.vectorized import VectorizedNetwork

net = VectorizedNetwork(max_neurons=20000)
net.init_neurons(10000)
net.connect_knn(k=4)
for t in range(1, 1001):
    net.tick(t)
```

也可以通过 CLI 基准测试评估性能：
```bash
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 200
```

## 3. 什么是 Prediction Error (PE)？

PE 衡量网络对下一个信号的预测准确度。每个神经元维护一个转移计数表，用于预测最可能的下一个 token。当实际 token 意外出现时，PE 值较高。

PE 的趋势（通过 `pe_moving_avg`）是观察网络学习的关键指标：
- PE 持续下降 → 网络正在学习输入模式
- PE 保持高位 → 输入过于随机或网络结构不合理

## 4. 如何复现实验？

使用 `seed_all()` 确保完全可复现：

```python
from neurogenesis_lab.config import seed_all
seed_all(42)  # 统一设置 random + numpy 的种子
```

CLI 方式：
```bash
python -m neurogenesis_lab run --seed 42
```

## 5. 输出文件在哪里？

默认输出在 `outputs/` 目录下，按实验名和时间戳自动分目录：

```
outputs/
├── run_20250101_120000/
│   └── run.jsonl
├── minimal_20250101_120100/
│   └── run.jsonl
└── evolution_20250101_120200/
    └── run.jsonl
```

每个 JSONL 文件包含：
1. `config` 记录（实验配置）
2. 每个 tick 的指标快照
3. `summary` 记录（实验总结）

## 6. Schema Version 有什么用？

`SCHEMA_VERSION = "2.0.0"` 嵌入在所有输出记录中，用于：
- 标识输出格式版本
- 下游分析工具可据此选择解析逻辑
- 跨版本实验结果对比时的兼容性判断

## 7. 三种神经元模型有何区别？

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| `Neuron` | 基础 IF 模型，线性动力学 | 教学、快速原型 |
| `LIFNeuron` | 漏整合放电 (Leaky IF)，指数衰减 | 标准仿真 |
| `IzhikevichNeuron` | 二维动力学 (v, u)，可模拟多种放电模式 | 复杂行为研究 |
| `AdExNeuron` | 自适应指数 IF，最接近生物真实 | 高保真度仿真 |

## 8. 如何添加自定义神经元模型？

继承 `Neuron` 并重写 `update()` 方法：

```python
from neurogenesis_lab.neuron import Neuron, safe_float

class MyNeuron(Neuron):
    def update(self, dt: float = 1.0) -> bool:
        I = safe_float(sum(self._input_buffer))
        self._input_buffer.clear()
        self.v = safe_float(self.v + I * dt)
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.spike_count += 1
            return True
        return False
```

## 9. `congestion_penalty` 是怎么工作的？

当一个神经元繁殖时，如果其连接数超过 10，繁殖概率会乘以惩罚因子：

```
effective_threshold = reproduction_threshold * (1 + congestion_penalty * (links - 10))
```

这防止少数"超级节点"垄断繁殖资源。

## 10. 如何对比两组实验？

使用对照实验模板：
```bash
python examples/experiment_contrast.py
```

或手动对比：
```python
from neurogenesis_lab.config import SimConfig, seed_all

# 两组配置只改变一个变量
cfg_a = SimConfig(seed=42, network=NetworkParams(mutation_scale=0.02))
cfg_b = SimConfig(seed=42, network=NetworkParams(mutation_scale=0.10))

# 分别运行，对比输出
```

关键：保持相同的 `seed` 确保除控制变量外的其他条件一致。
