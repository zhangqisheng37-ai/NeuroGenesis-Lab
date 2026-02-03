import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder
from neurogenesis_lab.neuron import Neuron


def main() -> None:
    encoder = Encoder()
    neuron = Neuron()
    expr = "1+1=2"
    events = encoder.to_events(encoder.encode_expr(expr), src_id=neuron.id)
    for event in events:
        neuron.receive(event)
    stats = neuron.observe()
    print("events")
    for event in events:
        print(asdict(event))
    print("observe")
    print(stats)


if __name__ == "__main__":
    main()
