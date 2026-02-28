# neuron — 神经元模型

多模型脉冲神经元实现，采用组件化架构（NeuronDynamics / NeuronEvolution / NeuronMemory）。

## 神经元类层次

- **`Neuron`** — 基类（dataclass），提供能量、记忆、位移、适应度与繁殖机制
- **`LIFNeuron(Neuron)`** — Leaky Integrate-and-Fire 标准模型
- **`IzhikevichNeuron(Neuron)`** — Izhikevich 模型（v²+5v+140 动力学）
- **`AdExNeuron(Neuron)`** — Adaptive Exponential 模型（指数项+适应电流）

## 组件

- **`NeuronDynamics`** — 动力学状态（v, u, w, v_rest, v_reset, v_th, tau_m 等）
- **`NeuronEvolution`** — 演化统计（spike_count, propagation_count, prediction_hits）
- **`NeuronMemory`** — 记忆管理（mem_size, mem deque, transitions Counter）

## 工具函数

- `clamp(value, minimum, maximum)` — 数值钳制
- `safe_float(value, fallback)` — NaN/Inf 安全转换
- `calculate_reward(novelty, energy_loss)` — 内在奖励计算

::: neurogenesis_lab.neuron
