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

# 输出治理（每10 tick采样 + gzip + 轮转）
python -m neurogenesis_lab run --log-interval 10 --gzip-output --max-file-size 5000000

# 性能基准测试
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100
```

## 实验模板

项目提供四个即开即用的实验脚本：

| 模板 | 用途 | 命令 |
|------|------|------|
| `experiment_minimal.py` | 快速验证 (50 ticks, 10 neurons) | `python examples/experiment_minimal.py` |
| `experiment_contrast.py` | 对照实验 (低突变 vs 高突变, 200 ticks) | `python examples/experiment_contrast.py` |
| `experiment_evolution.py` | 长程演化 (500 ticks, 20 neurons) | `python examples/experiment_evolution.py` |
| `experiment_genesis.py` | 创世纪实验 (500 ticks, 30 neurons) | `python examples/experiment_genesis.py` |

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
from neurogenesis_lab.learning import RuleComposer, RuleDiscovery, PatternMiner, QueryEngine

# 规则发现（基于 token 序列样本）
samples = [
    [ord("1"), ord("+"), ord("1"), ord("="), ord("2")],
    [ord("1"), ord("+"), ord("2"), ord("="), ord("3")],
    [ord("2"), ord("+"), ord("2"), ord("="), ord("4")],
]
store = RuleDiscovery(min_support=1).discover(samples)
composer = RuleComposer(store)
filled = composer.apply([ord("1"), ord("+"), ord("2"), ord("="), -1], missing_token=-1)
print("".join(chr(token) for token in filled))

# 表达式模式学习 + 查询
miner = PatternMiner()
miner.fit(["1+1=2", "1+2=3", "2+2=4"])
engine = QueryEngine(miner)
print(engine.query("1+2=?"))
print(engine.query_with_explanation("2+2=?"))
```

## 输出格式

所有输出均为 JSONL 格式，每行一个 JSON 对象（启用 `--gzip-output` 时为 `.jsonl.gz` 压缩文件）：

```json
{"type": "config", "schema_version": "2.0.0", "ticks": 100, ...}
{"schema_version": "2.0.0", "metrics": {...}, "learning": {...}, "trends": {...}}
{"type": "summary", "schema_version": "2.0.0", "total_ticks": 100, ...}
```

输出文件自动按实验名和时间戳分目录存放在 `outputs/` 下。

### 查看 `.jsonl.gz` 内容

如果开启了 `--gzip-output`，可用批量解压脚本生成可直接打开的 `.jsonl`：

```bash
# 解压单次实验目录
python examples/decompress_outputs.py --path outputs/<run_id>

# 递归解压整个 outputs
python examples/decompress_outputs.py --path outputs --recursive
```

## 下一步

- 参阅 [参数说明 (Parameters)](parameters.md) 了解所有可配置参数
- 参阅 [架构设计 (Architecture)](architecture.md) 理解系统内部工作原理
- 参阅 [可观测性指标 (Metrics)](metrics.md) 查看数据字段说明
- 参阅 [风险清单 (Risk List)](risk_list.md) 了解当前已知风险

## 实验有效激活检查（推荐）

运行：

```bash
python -m neurogenesis_lab run --ticks 100 --seed 42
```

至少确认以下三项：

1. 控制台总结中 `总放电 > 0` 且 `总传播 > 0`
2. 输出 JSONL 的 `metrics.alive_neurons` 在时间上有变化（不是全程恒定）
3. 输出 JSONL 的 `learning.total_spikes` 随 tick 增长

若未激活（例如放电/传播为 0），优先尝试：

- 提高 `--input-weight`（如 60~120）
- 降低 `[neuron].v_th`
- 提高 `[network].input_current_scale`
