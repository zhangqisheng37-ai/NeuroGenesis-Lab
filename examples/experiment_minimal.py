"""实验模板 — 最小实验

快速验证流程：创建小网络 → 注入信号 → 跑 50 ticks → 输出总结。
运行: python examples/experiment_minimal.py
"""

from pathlib import Path

from neurogenesis_lab.config import SimConfig, seed_all
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.observability import NetworkObserver, create_experiment_logger


def main() -> None:
    cfg = SimConfig(ticks=50, seed=42)
    seed_all(cfg.seed)

    # 构建 10 个 LIF 神经元
    neurons = [LIFNeuron() for _ in range(10)]
    network = Network(
        neurons={n.id: n for n in neurons},
        max_neurons=20,
        max_links=100,
    )
    # 环形连接
    for i, neuron in enumerate(neurons):
        network.connect(neuron.id, neurons[(i + 1) % len(neurons)].id, weight=1.0, delay=1)

    # 注入简单表达式
    encoder = Encoder()
    tokens = encoder.encode_expr("1+1=2")
    events = encoder.to_events(tokens, src_id=neurons[0].id)
    network.enqueue(events)

    # 输出
    logger = create_experiment_logger(Path("outputs"), "minimal")
    observer = NetworkObserver(network)
    logger.log_config(cfg.to_dict())

    # 运行
    for t in range(1, cfg.ticks + 1):
        snapshot = network.tick(t, dt=cfg.dt)
        logger.log_tick(observer, t, snapshot)

    final = observer.get_network_metrics(cfg.ticks)
    logger.log_summary(cfg.ticks, final)
    summary = network.get_summary()

    print(f"[最小实验] 完成 {cfg.ticks} ticks")
    print(f"  存活: {summary['alive_neurons']}/{summary['total_neurons']}")
    print(f"  总放电: {summary['total_spikes']}")
    print(f"  输出: {logger.get_output_path()}")


if __name__ == "__main__":
    main()
