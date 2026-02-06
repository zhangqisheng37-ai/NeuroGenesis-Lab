import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.network import Network
from neurogenesis_lab.observability import JsonlWriter


def main() -> None:
    encoder = Encoder()
    neurons = [LIFNeuron() for _ in range(6)]
    network = Network(neurons={neuron.id: neuron for neuron in neurons}, max_neurons=24, max_links=80)

    for index, neuron in enumerate(neurons):
        target = neurons[(index + 1) % len(neurons)]
        network.connect(neuron.id, target.id, weight=1.0, delay=1)

    tokens = encoder.encode_expr("1+1=2")
    records: list[dict] = []
    neuron_counts: list[int] = []
    corpse_counts: list[int] = []
    for tick in range(1, 80):
        if tick % 4 == 0:
            source = neurons[tick % len(neurons)]
            base_events = encoder.to_events(tokens, src_id=source.id)
            scheduled = [
                Event(token_id=event.token_id, t=event.t + tick, src_id=event.src_id, meta=event.meta)
                for event in base_events
            ]
            network.enqueue(scheduled)
        snapshot = network.tick(tick)
        record = {
            **snapshot,
            "total_neurons": len(network.neurons),
            "corpse_count": len(network.corpses),
            "queue_size": len(network.queue),
        }
        records.append(record)
        neuron_counts.append(record["total_neurons"])
        corpse_counts.append(record["corpse_count"])

    avg_neurons = sum(neuron_counts) / len(neuron_counts) if neuron_counts else 0.0
    records.append(
        {
            "type": "summary",
            "ticks": len(records),
            "min_neurons": min(neuron_counts) if neuron_counts else 0,
            "max_neurons": max(neuron_counts) if neuron_counts else 0,
            "avg_neurons": round(avg_neurons, 4),
            "final_neurons": neuron_counts[-1] if neuron_counts else 0,
            "final_corpse_count": corpse_counts[-1] if corpse_counts else 0,
        }
    )

    output_path = ROOT / "jsonl" / "run_evolution.jsonl"
    JsonlWriter(output_path).write(records)
    print(output_path)


if __name__ == "__main__":
    main()
