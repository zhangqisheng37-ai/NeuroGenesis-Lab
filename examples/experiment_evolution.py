"""实验模板 — 长程演化

500 ticks 长程演化实验，观察:
  1. 规则涌现 (RuleDiscovery)
  2. 种群稳定性 (存活/死亡平衡)
  3. PE 收敛趋势
运行: python examples/experiment_evolution.py
"""

from pathlib import Path

from neurogenesis_lab.config import SimConfig, NeuronParams, NetworkParams, EvolutionParams, seed_all
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.learning import RuleDiscovery, RuleStore
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron, IzhikevichNeuron, AdExNeuron
from neurogenesis_lab.observability import NetworkObserver, create_experiment_logger


def main() -> None:
    cfg = SimConfig(
        ticks=500,
        seed=42,
        neuron=NeuronParams(v_th=-48.0, spike_cost=0.04, input_gain=0.06, mem_size=128),
        network=NetworkParams(
            max_neurons=80,
            max_links=400,
            reproduction_threshold=2.0,
            energy_decay=0.003,
            corpse_ttl=80,
            move_cost=0.005,
        ),
        evolution=EvolutionParams(
            congestion_penalty=0.15,
            min_neurons=5,
            fitness_weights={"propagation": 1.0, "energy_efficiency": 0.8, "prediction_hit": 1.0},
        ),
    )
    seed_all(cfg.seed)

    # 20 个混合型神经元
    neuron_types = [LIFNeuron, IzhikevichNeuron, AdExNeuron]
    neurons = [neuron_types[i % len(neuron_types)]() for i in range(20)]
    network = Network(
        neurons={n.id: n for n in neurons},
        max_neurons=cfg.network.max_neurons,
        max_links=cfg.network.max_links,
        reproduction_threshold=cfg.network.reproduction_threshold,
        mutation_scale=cfg.network.mutation_scale,
    )
    for i, neuron in enumerate(neurons):
        for offset in [1, 2]:
            target = neurons[(i + offset) % len(neurons)]
            network.connect(neuron.id, target.id, weight=1.0, delay=offset)

    # 多组表达式注入，模拟持续学习
    encoder = Encoder()
    expressions = ["1+1=2", "2+3=5", "3-1=2", "2*3=6", "5+5=10"]
    for idx, expr in enumerate(expressions):
        tokens = encoder.encode_expr(expr)
        events = encoder.to_events(tokens, src_id=neurons[idx % len(neurons)].id)
        scheduled = [
            Event(token_id=e.token_id, t=e.t + idx * 20, src_id=e.src_id, meta=e.meta)
            for e in events
        ]
        network.enqueue(scheduled)

    # 规则发现引擎 + 样本收集
    discovery = RuleDiscovery()
    current_store = RuleStore()
    sample_buffer: list[list[int]] = []

    logger = create_experiment_logger(Path("outputs"), "evolution")
    observer = NetworkObserver(
        network,
        debug_neuron_ids=[neurons[0].id],
        neuron_filter=lambda n: n.energy > 3.0,  # 追踪高能神经元
    )
    logger.log_config(cfg.to_dict())

    # 运行
    checkpoint_interval = 100
    for t in range(1, cfg.ticks + 1):
        snapshot = network.tick(t, dt=cfg.dt)
        logger.log_tick(observer, t, snapshot)

        # 定期喂数据给规则发现引擎
        if t % 50 == 0:
            for n in list(network.neurons.values())[:5]:
                seq = [event.token_id for event in list(n.mem)[-10:]]
                if len(seq) >= 3:
                    sample_buffer.append(seq)
            if sample_buffer:
                current_store = discovery.discover(sample_buffer)

        # checkpoint 输出
        if t % checkpoint_interval == 0:
            summary = network.get_summary()
            trend = observer.get_trend_metrics(t, snapshot)
            rules = current_store.top_candidates_next(left=1, k=3)
            print(f"[tick {t}/{cfg.ticks}]")
            print(f"  存活: {summary['alive_neurons']}/{summary['total_neurons']}")
            print(f"  PE 均值: {trend['pe_moving_avg']:.4f}")
            print(f"  出生率: {trend['birth_rate']:.2f}, 死亡率: {trend['death_rate']:.2f}")
            print(f"  规则候选: {rules}")

    final = observer.get_network_metrics(cfg.ticks)
    logger.log_summary(cfg.ticks, final)
    summary = network.get_summary()

    print(f"\n{'='*50}")
    print(f"[长程演化] 完成 {cfg.ticks} ticks")
    print(f"  最终存活: {summary['alive_neurons']}/{summary['total_neurons']}")
    print(f"  总放电: {summary['total_spikes']}, 总传播: {summary['total_propagations']}")
    print(f"  出生: {summary['birth_count']}, 死亡: {summary['death_count']}")
    print(f"  输出: {logger.get_output_path()}")


if __name__ == "__main__":
    main()
