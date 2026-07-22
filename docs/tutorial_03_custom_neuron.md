# 教程 03：自定义神经元模型

NeuroGenesis Lab 提供了灵活的神经元接口，允许你轻松实现自定义的神经元动力学模型。

## 1. 神经元基类

所有神经元模型都继承自 `src/neurogenesis_lab/neuron.py` 中的 `Neuron` 基类。`Neuron` 是一个 `dataclass`，内部采用组件化架构（`NeuronDynamics` / `NeuronEvolution` / `NeuronMemory`），通过 `ComponentMixin`（`src/neurogenesis_lab/_components.py`）实现字段透明代理。

建议优先参考现有实现（`LIFNeuron` / `IzhikevichNeuron` / `AdExNeuron`）保持接口一致，尤其是：

- `update(dt, input_current)`：返回是否放电
- `emit(event, current_tick)`：按连接生成传播事件
- `spawn(mutation_scale, share)`：繁殖子代
- `fitness(weights)`：计算加权适应度

```python
@dataclass
class Neuron:
    id: str = ""           # UUID hex
    pos: tuple = (0.0, 0.0)  # 空间坐标
    energy: float = 1.0    # 当前能量
    alive: bool = True     # 是否存活
    links: dict = ...      # 出边字典
    states: dict = ...     # 内部状态 (energy/entropy/emotion/novelty)
    # 内部组件：_dynamics, _evolution, _memory
    # 通过 ComponentMixin 代理访问 v, u, w, v_rest, tau_m 等
```

## 2. 实现自定义模型

要创建一个新的神经元模型，你需要继承 `Neuron`（dataclass）并重写 `update` 方法。

### 示例：实现一个简单的 Integrate-and-Fire (IF) 神经元

IF 神经元没有泄漏（Leaky）机制，只要输入电流足够，膜电位就会持续上升直到达到阈值。

```python
from neurogenesis_lab.neuron import Neuron

class IFNeuron(Neuron):
    def __init__(self, neuron_id: str | None = None, **kwargs):
        super().__init__(neuron_id, **kwargs)
        self.v_rest = kwargs.get("v_rest", -65.0)
        self.v_reset = kwargs.get("v_reset", -65.0)
        self.v_th = kwargs.get("v_th", -50.0)
        self.r_m = kwargs.get("r_m", 1.0)
        self.input_gain = kwargs.get("input_gain", 0.1)

    def update(self, dt: float = 1.0, input_current: float = 0.0) -> bool:
        """
        更新膜电位并返回是否放电
        """
        # 简单的积分：dV = I * R * dt
        dv = i_inj * self.r_m * self.input_gain * dt
        self.v += dv

        # 检查是否达到阈值
        if self.v >= self.v_th:
            self.v = self.v_reset
            return True
        return False
```

> 说明：上述示例用于展示动力学思路。若要完整接入当前引擎，请对齐 `Neuron` 基类的实际更新入口与事件/能量字段处理逻辑。

## 3. 在网络中使用自定义神经元

你可以将自定义神经元传递给 `Network` 的初始化函数。

```python
from neurogenesis_lab.network import Network
from my_neurons import IFNeuron

# 创建 10 个 IF 神经元
neurons = [IFNeuron() for _ in range(10)]
neuron_dict = {n.id: n for n in neurons}

# 初始化网络
network = Network(neurons=neuron_dict)

# 运行仿真
for t in range(1, 101):
    network.tick(t)
```

建议先用短程冒烟验证：

```bash
python -m neurogenesis_lab run --ticks 100 --visualize
```

重点观察是否出现：

1. `总放电 > 0`
2. `总传播 > 0`
3. 存活数量随时间变化（非恒定）

## 4. 演化与变异

如果你希望你的自定义神经元支持演化（参数变异），你需要重写 `mutate` 方法。

```python
    def mutate(self, scale: float = 0.1) -> None:
        super().mutate(scale)
        # 变异特定参数
        import random
        if random.random() < scale:
            self.input_gain *= random.uniform(0.9, 1.1)
```

> **注意**：v0.6.0 的 `Neuron` 基类采用组件化架构（`NeuronDynamics` / `NeuronEvolution` / `NeuronMemory`），通过 `ComponentMixin`（`src/neurogenesis_lab/_components.py`）代理访问。自定义子类的额外字段会保留在子类 `__dict__` 中，不与组件冲突。遗传属性由 `config.py` 中的 `HERITABLE_ATTRS` 常量统一管理。

若需要让自定义模型被 Rust 路径加速，推荐方式是：

1. 在 Python 侧先确认模型行为稳定（统计指标正确）
2. 将最耗时内核抽离为纯数值步骤
3. 在 `rust_src/neurogenesis_core` 中新增 PyO3 导出函数，并在 `core_bridge.py` 提供 fallback 包装

## 5. 总结

通过继承 `Neuron` 类，你可以实现各种复杂的神经元模型（如 Hodgkin-Huxley、FitzHugh-Nagumo 等），并接入 NeuroGenesis Lab 的仿真与观测框架。

完成模型接入后，建议使用以下闭环检查：

```bash
python -m neurogenesis_lab info
python -m neurogenesis_lab run --ticks 100 --seed 42
python -m neurogenesis_lab benchmark --neurons 1000 --ticks 50 --rust-size 5000
pytest
```
