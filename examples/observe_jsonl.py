import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.network import Network
from neurogenesis_lab.observability import JsonlWriter


def main() -> None:
    encoder = Encoder()
    neuron_a = LIFNeuron()
    neuron_b = LIFNeuron()
    network = Network(neurons={neuron_a.id: neuron_a, neuron_b.id: neuron_b})
    network.connect(neuron_a.id, neuron_b.id, weight=1.0, delay=1)

    expr = "1+1=2"
    events = encoder.to_events(encoder.encode_expr(expr), src_id=neuron_a.id)
    network.enqueue(events)

    records: list[dict] = []
    for tick in range(1, 8):
        snapshot = network.tick(tick)
        avg_energy = sum(neuron.energy for neuron in network.neurons.values()) / len(
            network.neurons
        )
        avg_novelty = sum(neuron.states["novelty"] for neuron in network.neurons.values()) / len(
            network.neurons
        )
        avg_emotion = sum(neuron.states["emotion"] for neuron in network.neurons.values()) / len(
            network.neurons
        )
        record = {
            **snapshot,
            "queue_size": len(network.queue),
            "avg_energy": round(avg_energy, 4),
            "avg_novelty": round(avg_novelty, 4),
            "avg_emotion": round(avg_emotion, 4),
        }
        records.append(record)

    output_path = ROOT / "jsonl" / "run.jsonl"
    JsonlWriter(output_path).write(records)
    print(output_path)


if __name__ == "__main__":
    main()
