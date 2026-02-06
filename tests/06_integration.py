import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.learning import MemoryBank, PatternMiner, QueryEngine
from neurogenesis_lab.neuron import AdExNeuron, IzhikevichNeuron, LIFNeuron
from neurogenesis_lab.network import Network
from neurogenesis_lab.observability import JsonlWriter, NetworkObserver, ObservabilityLogger


def main() -> None:
    encoder = Encoder()
    neurons = [LIFNeuron(), IzhikevichNeuron(), AdExNeuron(), LIFNeuron()]
    network = Network(neurons={neuron.id: neuron for neuron in neurons}, max_neurons=32, max_links=120)

    for index, neuron in enumerate(neurons):
        target = neurons[(index + 1) % len(neurons)]
        network.connect(neuron.id, target.id, weight=1.0, delay=1)
    network.connect(neurons[0].id, neurons[2].id, weight=0.6, delay=2)

    tokens = encoder.encode_expr("1+2=3")
    for tick in (1, 6, 12):
        source = neurons[tick % len(neurons)]
        base_events = encoder.to_events(tokens, src_id=source.id)
        scheduled = [
            Event(token_id=event.token_id, t=event.t + tick, src_id=event.src_id, meta=event.meta)
            for event in base_events
        ]
        network.enqueue(scheduled)

    output_path = ROOT / "jsonl" / "run_integration.jsonl"
    observer = NetworkObserver(network, debug_neuron_ids=[neurons[0].id, neurons[1].id])
    logger = ObservabilityLogger(output_path, debug_mode=True)

    for tick in range(1, 21):
        network.tick(tick)
        logger.log_tick(observer, tick)

    final_metrics = observer.get_tick_metrics(20)
    logger.log_summary(20, final_metrics)

    bank = MemoryBank()
    bank.add_many(["1+1=2", "1+2=3", "2+2=4"])
    miner = PatternMiner()
    miner.fit(bank.expressions)
    engine = QueryEngine(miner=miner)

    writer = JsonlWriter(output_path)
    queries = ["1+2=?", "2+2=?", "3+3=?"]
    known = 0
    for query in queries:
        answer = engine.query(query)
        if answer != "unknown":
            known += 1
        writer.append({"type": "learning", "query": query, "answer": answer})
    writer.append(
        {
            "type": "learning_summary",
            "training_samples": len(bank.expressions),
            "queries": len(queries),
            "known": known,
        }
    )

    print(output_path)


if __name__ == "__main__":
    main()
