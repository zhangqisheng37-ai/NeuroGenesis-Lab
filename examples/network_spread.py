import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder
from neurogenesis_lab.neuron import Neuron
from neurogenesis_lab.network import Network


def main() -> None:
    encoder = Encoder()
    neuron_a = Neuron()
    neuron_b = Neuron()
    network = Network(neurons={neuron_a.id: neuron_a, neuron_b.id: neuron_b})
    network.connect(neuron_a.id, neuron_b.id, weight=1.0, delay=1)

    expr = "1+1=2"
    events = encoder.to_events(encoder.encode_expr(expr), src_id=neuron_a.id)
    network.enqueue(events)

    for tick in range(1, 8):
        snapshot = network.tick(tick)
        print(snapshot)


if __name__ == "__main__":
    main()
