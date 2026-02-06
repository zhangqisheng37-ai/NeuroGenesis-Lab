import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder
from neurogenesis_lab.neuron import AdExNeuron, IzhikevichNeuron, LIFNeuron
from neurogenesis_lab.network import Network
from neurogenesis_lab.observability import JsonlWriter, NetworkObserver


def main() -> None:
    encoder = Encoder()
    neuron_a = LIFNeuron()
    neuron_b = IzhikevichNeuron()
    neuron_c = AdExNeuron()
    network = Network(
        neurons={
            neuron_a.id: neuron_a,
            neuron_b.id: neuron_b,
            neuron_c.id: neuron_c,
        }
    )
    network.connect(neuron_a.id, neuron_b.id, weight=1.0, delay=1)
    network.connect(neuron_b.id, neuron_c.id, weight=1.0, delay=1)
    network.connect(neuron_c.id, neuron_a.id, weight=1.0, delay=1)

    expr = "11111111"
    events = encoder.to_events(encoder.encode_expr(expr), src_id=neuron_a.id)
    network.enqueue(events)

    observer = NetworkObserver(network)
    records: list[dict] = []

    for tick in range(1, 12):
        snapshot = network.tick(tick)
        metrics = observer.get_tick_metrics(tick)
        records.append({"snapshot": snapshot, "metrics": metrics})

    final_metrics = observer.get_tick_metrics(11)
    records.append({"type": "summary", "ticks": 11, "final_metrics": final_metrics})

    output_path = ROOT / "jsonl" / "run_network_spread.jsonl"
    JsonlWriter(output_path).write(records)
    print(output_path)


if __name__ == "__main__":
    main()
