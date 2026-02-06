import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.network import Network


def test_reproduction_happens_when_energy_above_threshold() -> None:
    parent = LIFNeuron(energy=3.2)
    network = Network(
        neurons={parent.id: parent},
        reproduction_threshold=2.5,
        reproduction_share=0.4,
        max_neurons=8,
    )

    snapshot = network.tick(1)

    assert snapshot["alive_neurons"] >= 1
    assert len(network.neurons) >= 2


def test_death_removes_neuron_and_creates_corpse() -> None:
    doomed = LIFNeuron(energy=0.0)
    network = Network(neurons={doomed.id: doomed}, max_neurons=8)

    network.tick(1)

    assert len(network.neurons) == 0
    assert len(network.corpses) == 1
