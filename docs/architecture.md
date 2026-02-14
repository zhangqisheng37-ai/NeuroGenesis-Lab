# 系统架构 (System Architecture)

本文档详细描述 NeuroGenesis Lab 的核心架构与设计原理。系统旨在构建一个基于生物动力学约束的数字生命演化平台。

## 1. 核心设计理念

系统遵循以下核心设计原则：

1.  **事件驱动 (Event-Driven)**：所有信息交互通过离散的 `Event` 对象进行，支持精确的时序控制与延迟模拟。
2.  **能量约束 (Energy Constraint)**：任何活动（放电、移动、繁殖）都消耗能量，能量是生存与演化的核心货币。
3.  **双重动力学 (Dual Dynamics)**：
    -   **快系统 (Fast)**：毫秒级的膜电位变化 (LIF 模型) 与脉冲传播。
    -   **慢系统 (Slow)**：代际的突变、选择与神经网络结构的重组。
4.  **可观测性 (Observability)**：系统内部状态对外部完全透明，支持“白盒”实验分析。

## 2. 系统全景图

```mermaid
graph TD
    User[用户/实验脚本] --> Config[SimConfig]
    Config --> Network[Network (演化引擎)]
    
    subgraph "Simulation Loop (Tick)"
        Network --> Queue[EventQueue (优先队列)]
        Queue --> |Pop Events| Neurons[Neuron Population]
        Neurons --> |Spike/Emit| Queue
        
        Network --> |Evolution| Scavenger[尸体回收]
        Network --> |Evolution| Reproduction[繁殖系统]
    end
    
    Neurons --> |State| Observability[Observability System]
    Observability --> |JSONL| Storage[磁盘存储]
    
    subgraph "Neuron Internals"
        LIF[LIF 动力学]
        Memory[短期记忆 (Deque)]
        RuleDiscovery[规则发现 (Learning)]
    end
```

## 3. 核心组件详解

### 3.1 仿真循环 (The Simulation Loop)

仿真以离散时间步 `tick` 为单位推进。每个 `tick` (默认 dt=1.0ms) 包含以下阶段：

1.  **事件分发 (Event Dispatch)**：
    -   从 `EventQueue` 中取出所有 `t <= current_tick` 的事件。
    -   将事件路由到目标神经元。
2.  **神经元更新 (Neuron Update)**：
    -   计算膜电位变化 `dv/dt` (LIF 公式)。
    -   判断是否达到阈值 `v_th`。
    -   如果放电 (Spike)，生成新的输出事件。
3.  **网络演化 (Network Evolution)**：
    -   **能量衰减**：全局扣除基础代谢能量。
    -   **死亡判定**：能量 <= 0 的神经元转化为 `Corpse`。
    -   **繁殖判定**：能量 > `reproduction_threshold` 的神经元进行分裂。
    -   **尸体回收**：存活神经元若在尸体 `scavenger_radius` 范围内，可回收部分能量。
4.  **数据采集 (Observability)**：
    -   收集全网快照（存活数、总能量、预测误差等）。
    -   写入日志。

### 3.2 神经元模型 (Neuron Model)

采用 **Leaky Integrate-and-Fire (LIF)** 模型，并增加了能量与认知维度：

#### 物理层 (Physics)
$$ \tau_m \frac{dV}{dt} = -(V - E_{rest}) + R_m I(t) $$

- **输入电流 $I(t)$**：来自外部输入事件和突触前神经元的脉冲。
- **不应期**：放电后电位重置为 $V_{reset}$。

#### 能量层 (Energy)
$$ E(t+1) = E(t) - E_{decay} - E_{spike} \cdot \mathbb{I}(spike) + E_{gain} \cdot I_{in} $$

- 能量耗尽导致死亡。
- 能量富余触发繁殖。

#### 认知层 (Cognition)
- **Memory**：维护一个固定长度的 `Event` 队列，记录最近接收到的 Token 序列。
- **Transitions**：统计 Token 转移概率（如 A -> B），用于简单的预测与规则学习。
- **Novelty**：基于短期记忆计算输入的新颖性，作为内在奖励信号。

### 3.3 事件系统与时间语义 (Event System & Time Semantics)

为保证仿真的严谨性，系统定义了明确的时间语义：

1.  **Current Tick ($t$)**：仿真器的当前离散时刻。
2.  **Event Time ($t_{event}$)**：事件发生的时刻。
3.  **Propagation Delay ($d$)**：信号传输的耗时。

**时序约束**：
- **输入事件**：外部输入事件可提前入队；当仿真时钟满足 `current_tick >= t_event` 时会被消费。
- **传播机制**：
    - 源神经元在 $t$ 时刻放电。
    - 产生输出事件 $E$。
    - $E.t = t + d$。
    - 事件 $E$ 被插入优先队列，只有当仿真时钟推进到 $t + d$ 时，目标神经元才会收到该事件。
    - 若 $d=0$，则事件在 **下一个** tick ($t+1$) 处理（模拟最小突触延迟），防止因果循环。

事件是网络中信息传递的唯一载体：

```python
@dataclass
class Event:
    token_id: int   # 携带的信息 (Token)
    t: int          # 发生时间 (Tick)
    src_id: str     # 路由目标神经元 ID
    meta: dict      # 附加信息（可包含来源 from / 权重 weight）
```

支持 **延迟 (Delay)**：
- 当神经元 A 在 $t$ 时刻向 B 发送脉冲，且连接延迟为 $d$。
- 事件会被插入队列的 $t + d$ 时刻。
- 这种机制允许模拟长程连接和时序逻辑。

## 4. 演化机制 (Evolutionary Mechanics)

系统通过局部相互作用实现全局演化：

### 4.1 繁殖 (Reproduction)
- **触发**：神经元能量累积超过阈值。
- **过程**：
    - 母体消耗能量。
    - 创建子代神经元，位置在母体附近。
    - **参数遗传**：子代继承母体参数（如 `v_th`, `tau_m`），并叠加高斯噪声突变。
    - **连接继承**：子代以一定概率继承母体的连接结构。

### 4.2 尸体回收 (Scavenging)
- 死亡神经元变成 `Corpse` 对象，保留剩余能量的一小部分。
- 尸体能量随时间衰减。
- 附近的存活神经元可以“吞噬”尸体能量，模拟生态系统中的能量循环。

## 5. 学习与规则发现 (Learning)

NeuroGenesis Lab 探索“从脉冲到符号”的涌现：

1.  **Tokenization**：外部输入被编码为离散 Token。
2.  **Pattern Mining**：神经元记录 Token 共现频率。
3.  **Rule Formation**：高频转移（如 Token 1 总是跟随 Token 2）被固化为规则。
4.  **Prediction**：神经元尝试预测下一个 Token，预测准确度 (Prediction Hit) 是适应度函数的一部分，奖励具有预测能力的神经元。

---

*设计文档版本：v1.0 (对应代码 v0.2.0)*
