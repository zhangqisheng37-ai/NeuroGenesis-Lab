"""实验模板 — 对照实验

对比两组参数（低突变 vs 高突变）的演化表现。
运行: python examples/experiment_contrast.py
"""

from pathlib import Path

from neurogenesis_lab.config import SimConfig, NetworkParams, EvolutionParams, seed_all
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron, IzhikevichNeuron, AdExNeuron
from neurogenesis_lab.observability import NetworkObserver, create_experiment_logger


def run_trial(label: str, cfg: SimConfig, neuron_count: int = 15) -> dict:
    """运行一次实验并返回总结"""
    seed_all(cfg.seed)

    neuron_types = [LIFNeuron, IzhikevichNeuron, AdExNeuron]
    neurons = [neuron_types[i % len(neuron_types)]() for i in range(neuron_count)]
    network = Network(
        neurons={n.id: n for n in neurons},
        max_neurons=cfg.network.max_neurons,
        max_links=cfg.network.max_links,
        reproduction_threshold=cfg.network.reproduction_threshold,
        mutation_scale=cfg.network.mutation_scale,
    )
    for i, neuron in enumerate(neurons):
        network.connect(neuron.id, neurons[(i + 1) % len(neurons)].id, weight=1.0, delay=1)

    encoder = Encoder()
    for expr in ["1+2=3", "3-1=2", "2*3=6"]:
        tokens = encoder.encode_expr(expr)
        events = encoder.to_events(tokens, src_id=neurons[0].id)
        network.enqueue(events)

    logger = create_experiment_logger(Path("outputs"), f"contrast_{label}")
    observer = NetworkObserver(network)
    logger.log_config(cfg.to_dict())

    for t in range(1, cfg.ticks + 1):
        snapshot = network.tick(t, dt=cfg.dt)
        logger.log_tick(observer, t, snapshot)

    final = observer.get_network_metrics(cfg.ticks)
    logger.log_summary(cfg.ticks, final)
    summary = network.get_summary()
    summary["label"] = label
    summary["output"] = str(logger.get_output_path())
    return summary


def main() -> None:
    ticks = 200
    seed = 42

    # A组: 低突变
    cfg_a = SimConfig(
        ticks=ticks,
        seed=seed,
        network=NetworkParams(mutation_scale=0.02, max_neurons=50, max_links=200),
        evolution=EvolutionParams(mutation_amplitude=0.02, min_neurons=3),
    )
    # B组: 高突变
    cfg_b = SimConfig(
        ticks=ticks,
        seed=seed,
        network=NetworkParams(mutation_scale=0.10, max_neurons=50, max_links=200),
        evolution=EvolutionParams(mutation_amplitude=0.10, min_neurons=3),
    )

    print("=" * 50)
    print("[对照实验] 低突变 vs 高突变")
    print("=" * 50)

    result_a = run_trial("low_mutation", cfg_a)
    result_b = run_trial("high_mutation", cfg_b)

    print(f"\n{'指标':<24} {'低突变':>12} {'高突变':>12}")
    print("-" * 48)
    for key in ["alive_neurons", "total_spikes", "total_propagations", "birth_count", "death_count"]:
        print(f"{key:<24} {result_a[key]:>12} {result_b[key]:>12}")

    print(f"\n输出:")
    print(f"  A: {result_a['output']}")
    print(f"  B: {result_b['output']}")


if __name__ == "__main__":
    main()
