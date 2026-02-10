"""实验模板 — 创世纪 (Genesis)
旨在创造一个资源丰富、生存压力较小的环境，促进种群大爆发。

运行: python examples/experiment_genesis.py
"""

from pathlib import Path

from neurogenesis_lab.config import SimConfig, NeuronParams, NetworkParams, EvolutionParams, seed_all
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.learning import RuleDiscovery, RuleStore
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron, IzhikevichNeuron, AdExNeuron
from neurogenesis_lab.observability import NetworkObserver, create_experiment_logger


def main() -> None:
    # 调整上帝参数：创造伊甸园
    cfg = SimConfig(
        ticks=500,
        seed=42,
        neuron=NeuronParams(
            v_th=-50.0,        # 降低阈值，更容易兴奋 (-48 -> -50)
            spike_cost=0.01,   # 降低放电成本 (0.04 -> 0.01)
            input_gain=0.2,    # 提高输入敏感度 (0.06 -> 0.2)
            mem_size=128
        ),
        network=NetworkParams(
            max_neurons=200,   # 提高种群上限 (80 -> 200)
            max_links=1000,
            reproduction_threshold=1.5, # 降低繁殖门槛 (2.0 -> 1.5)
            energy_decay=0.001,         # 降低能量衰减 (0.003 -> 0.001)
            corpse_ttl=80,
            move_cost=0.001,            # 降低移动成本 (0.005 -> 0.001)
            mutation_scale=0.1          # 提高突变率，增加多样性
        ),
        evolution=EvolutionParams(
            congestion_penalty=0.05,    # 降低拥堵惩罚
            min_neurons=10,
            fitness_weights={"propagation": 1.5, "energy_efficiency": 0.5, "prediction_hit": 1.0},
        ),
    )
    seed_all(cfg.seed)

    # 初始种群：30 个混合型神经元 (增加初始数量)
    neuron_types = [LIFNeuron, IzhikevichNeuron, AdExNeuron]
    neurons = [neuron_types[i % len(neuron_types)]() for i in range(30)]
    
    # 初始化网络，给予所有神经元更高的初始能量
    for n in neurons:
        n.energy = 2.0  # 默认通常是 1.0
        
    network = Network(
        neurons={n.id: n for n in neurons},
        max_neurons=cfg.network.max_neurons,
        max_links=cfg.network.max_links,
        reproduction_threshold=cfg.network.reproduction_threshold,
        mutation_scale=cfg.network.mutation_scale,
        energy_decay=cfg.network.energy_decay,  # 显式传递
        move_cost=cfg.network.move_cost,        # 显式传递
    )
    
    # 建立更密集的初始连接
    for i, neuron in enumerate(neurons):
        # 连接到最近的 3 个邻居 (逻辑上的邻居)
        for offset in [1, 2, 3]:
            target = neurons[(i + offset) % len(neurons)]
            network.connect(neuron.id, target.id, weight=1.5, delay=offset) # 增加初始权重

    # 持续的“天降甘露” (持续的外部刺激注入)
    encoder = Encoder()
    expressions = ["1+1=2", "2+3=5", "3-1=2", "2*3=6", "5+5=10", "A->B", "B->C", "C->A"]
    
    # 扩展注入计划，覆盖整个时间轴
    for cycle in range(5): # 5 个循环，覆盖更多时间
        base_time = cycle * 100
        for idx, expr in enumerate(expressions):
            tokens = encoder.encode_expr(expr)
            # 随机选一个源神经元
            src_idx = (idx + cycle) % len(neurons)
            # 注意：如果该神经元已死亡，enqueue 会自动处理（network.tick 中会检查 src_id）
            # 但为了保险，我们只在初始列表中选，如果它死了，事件会被忽略
            events = encoder.to_events(tokens, src_id=neurons[src_idx].id)
            scheduled = [
                Event(token_id=e.token_id, t=e.t + base_time + idx * 10, src_id=e.src_id, meta=e.meta)
                for e in events
            ]
            network.enqueue(scheduled)

    # 规则发现引擎 + 样本收集
    discovery = RuleDiscovery()
    current_store = RuleStore()
    sample_buffer: list[list[int]] = []

    # 使用 "genesis" 作为实验名前缀
    logger = create_experiment_logger(Path("outputs"), "genesis")
    observer = NetworkObserver(
        network,
        debug_neuron_ids=[neurons[0].id],
    )
    logger.log_config(cfg.to_dict())

    print(f"开始创世纪实验... 参数: Decay={cfg.network.energy_decay}, SpikeCost={cfg.neuron.spike_cost}")

    # 运行
    checkpoint_interval = 50
    for t in range(1, cfg.ticks + 1):
        # --- 光合作用 (Photosynthesis) ---
        # 模拟环境背景能量，确保基础生存，让神经元专注于处理信息而不是仅仅为了活命
        for n in network.neurons.values():
            if n.alive:
                n.energy += 0.002 # 抵消掉 energy_decay (0.001) 还有富余
        
        # --- 随机背景噪声 (Background Noise) ---
        # 每 tick 都有概率随机刺激几个神经元，模拟热噪声
        import random
        if t % 2 == 0: # 每 2 ticks
            lucky_ones = random.sample(list(network.neurons.keys()), k=min(len(network.neurons), 3))
            noise_events = []
            for nid in lucky_ones:
                # 注入一个随机 token 事件
                noise_events.append(Event(token_id=random.randint(1, 100), t=t, src_id=nid, meta={"type": "noise"}))
            network.enqueue(noise_events)

        snapshot = network.tick(t, dt=cfg.dt)
        logger.log_tick(observer, t, snapshot)

        # 样本收集
        if t % 50 == 0:
            active_neurons = [n for n in network.neurons.values() if n.alive]
            # 从活跃神经元中采样
            for n in active_neurons[:10]:
                seq = [event.token_id for event in list(n.mem)[-10:]]
                if len(seq) >= 3:
                    sample_buffer.append(seq)
            if sample_buffer:
                current_store = discovery.discover(sample_buffer)
                sample_buffer = [] # 清空 buffer

        # checkpoint 输出
        if t % checkpoint_interval == 0:
            summary = network.get_summary()
            trend = observer.get_trend_metrics(t, snapshot)
            rules = current_store.top_candidates_next(left=1, k=3)
            
            print(f"[tick {t}/{cfg.ticks}]")
            print(f"  存活: {summary['alive_neurons']}/{summary['total_neurons']}")
            print(f"  出生: {trend['birth_rate']:.2f}, 死亡: {trend['death_rate']:.2f}")
            if rules:
                print(f"  规则候选: {rules}")

    print("\n" + "="*50)
    final_summary = network.get_summary()
    print(f"[创世纪] 完成 {cfg.ticks} ticks")
    print(f"  最终存活: {final_summary['alive_neurons']}/{final_summary['total_neurons']}")
    print(f"  总出生: {network.birth_count}, 总死亡: {network.death_count}")
    print(f"  输出: {logger.get_output_path()}")

if __name__ == "__main__":
    main()
