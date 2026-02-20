import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Event
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.network import Network


def test_spike_emits_events_and_uses_current_tick_for_delay() -> None:
    neuron_a = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
    neuron_b = LIFNeuron()
    network = Network(
        neurons={neuron_a.id: neuron_a, neuron_b.id: neuron_b},
        input_current_scale=5.0,
    )
    network.connect(neuron_a.id, neuron_b.id, weight=1.0, delay=2)

    network.enqueue([Event(token_id=ord("1"), t=1, src_id=neuron_a.id, meta={"weight": 1.0})])

    tick_1 = network.tick(1)
    assert tick_1["propagation_count"] > 0
    assert tick_1["prediction_error"] == 0.0
    assert len(network.queue._heap) > 0
    assert network.queue._heap[0][0] == 3

    tick_2 = network.tick(2)
    assert tick_2["ready_events"] == 0
    assert tick_2["prediction_error"] >= 0.0

    tick_3 = network.tick(3)
    assert tick_3["ready_events"] >= 1
    assert tick_3["prediction_error"] >= 0.0
