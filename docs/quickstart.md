# 快速开始

## 安装

```bash
pip install -e .
```

## 30 秒上手

```bash
# 查看版本和默认配置
python -m neurogenesis_lab info

# 运行最小实验（默认 100 ticks）
python -m neurogenesis_lab run

# 指定参数
python -m neurogenesis_lab run --ticks 200 --seed 123 --neurons 12

# 使用 TOML 配置文件
python -m neurogenesis_lab run --config config.toml

# 性能基准测试
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100
```

## 实验模板

项目提供三个即开即用的实验脚本：

| 模板 | 用途 | 命令 |
|------|------|------|
| `experiment_minimal.py` | 快速验证 (50 ticks, 10 neurons) | `python examples/experiment_minimal.py` |
| `experiment_contrast.py` | 对照实验 (低突变 vs 高突变) | `python examples/experiment_contrast.py` |
| `experiment_evolution.py` | 长程演化 (500 ticks, 20 neurons) | `python examples/experiment_evolution.py` |

## 项目结构

```
src/neurogenesis_lab/
├── config.py          # 集中配置系统 (SimConfig)
├── neuron.py          # 神经元模型 (LIF, Izhikevich, AdEx)
├── network.py         # 网络引擎 (事件队列, 演化)
├── encoding.py        # 编码系统 (Token, Event)
├── learning.py        # 学习引擎 (规则发现, 查询)
├── observability.py   # 可观察性 (指标, 日志, 实验管理)
├── vectorized.py      # 向量化引擎 (NumPy 加速)
├── logger.py          # 基础日志
└── __main__.py        # CLI 入口
```

## 代码示例

### 最小可运行代码

```python
from neurogenesis_lab.config import SimConfig, seed_all
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.network import Network

cfg = SimConfig(ticks=20, seed=42)
seed_all(cfg.seed)

neurons = [LIFNeuron() for _ in range(5)]
net = Network(neurons={n.id: n for n in neurons})
for i, n in enumerate(neurons):
    net.connect(n.id, neurons[(i + 1) % 5].id, weight=1.0, delay=1)

for t in range(1, cfg.ticks + 1):
    snap = net.tick(t)
    print(f"tick={t}, alive={snap['alive_neurons']}, PE={snap['prediction_error']:.4f}")
```

### 使用向量化引擎（大规模）

```python
import numpy as np
from neurogenesis_lab.vectorized import VectorizedNetwork

net = VectorizedNetwork(max_neurons=10000)
net.init_neurons(5000, energy_init=1.5)
net.connect_knn(k=4)

for t in range(1, 101):
    net.tick(t)

print(net.get_summary())
```

### 学习与规则发现

```python
from neurogenesis_lab.learning import RuleDiscovery, QueryEngine, PatternMiner

discovery = RuleDiscovery()
for token in [1, 2, 3, 1, 2, 3, 1, 2]:
    discovery.feed_token(token)

# 查询规则
print(discovery.store.top_candidates_next(top_k=5))

# 模式挖掘
miner = PatternMiner()
miner.feed([1, 2, 3, 1, 2, 3])
print(miner.top_patterns(k=3))
print(f"confidence(1,2,3) = {miner.confidence((1, 2, 3))}")
```

## 输出格式

所有输出均为 JSONL 格式，每行一个 JSON 对象：

```json
{"type": "config", "schema_version": "2.0.0", "ticks": 100, ...}
{"schema_version": "2.0.0", "metrics": {...}, "learning": {...}, "trends": {...}}
{"type": "summary", "schema_version": "2.0.0", "total_ticks": 100, ...}
```

输出文件自动按实验名和时间戳分目录存放在 `outputs/` 下。

## 下一步

- 参阅 [参数说明 (Parameters)](parameters.md) 了解所有可配置参数
- 参阅 [架构设计 (Architecture)](architecture.md) 理解系统内部工作原理
- 参阅 [可观测性指标 (Metrics)](metrics.md) 查看数据字段说明
- 参阅 [常见问题 (FAQ)](faq.md) 查看常见问题
