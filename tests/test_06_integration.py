import json

# Ensure src is in path for imports to work without package installation
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.learning import MemoryBank, PatternMiner, QueryEngine
from neurogenesis_lab.neuron import AdExNeuron, IzhikevichNeuron, LIFNeuron
from neurogenesis_lab.network import Network
from neurogenesis_lab.observability import JsonlWriter, NetworkObserver, ObservabilityLogger

def test_full_integration_pipeline(tmp_path):
    """
    Test the full pipeline:
    1. Network initialization with mixed neuron types
    2. Event encoding and scheduling
    3. Simulation execution (ticks)
    4. Observability logging
    5. Symbolic learning component
    """
    # 1. Setup Environment
    encoder = Encoder()
    neurons = [LIFNeuron(), IzhikevichNeuron(), AdExNeuron(), LIFNeuron()]
    # Use a dict for neurons as expected by Network
    neuron_map = {neuron.id: neuron for neuron in neurons}
    network = Network(neurons=neuron_map, max_neurons=32, max_links=120)

    # 2. Setup Connections
    # Connect in a ring
    for index, neuron in enumerate(neurons):
        target = neurons[(index + 1) % len(neurons)]
        network.connect(neuron.id, target.id, weight=1.0, delay=1)
    # Extra connection
    network.connect(neurons[0].id, neurons[2].id, weight=0.6, delay=2)

    # Verify initial state
    assert len(network.neurons) == 4
    assert network.neurons[neurons[0].id].links[neurons[1].id]["weight"] == 1.0

    # 3. Inject Events
    tokens = encoder.encode_expr("1+2=3")
    assert len(tokens) > 0
    
    # Schedule events at different ticks
    for tick in (1, 6, 12):
        source = neurons[tick % len(neurons)]
        base_events = encoder.to_events(tokens, src_id=source.id)
        scheduled = [
            Event(token_id=event.token_id, t=event.t + tick, src_id=event.src_id, meta=event.meta)
            for event in base_events
        ]
        network.enqueue(scheduled)

    assert len(network.queue) > 0

    # 4. Run Simulation
    # Use tmp_path for output to avoid cluttering the workspace
    output_path = tmp_path / "run_integration.jsonl"
    observer = NetworkObserver(network, debug_neuron_ids=[neurons[0].id, neurons[1].id])
    logger = ObservabilityLogger(output_path, debug_mode=True)

    total_ticks = 20
    for tick in range(1, total_ticks + 1):
        snapshot = network.tick(tick)
        logger.log_tick(observer, tick)
        
        # Verify snapshot structure
        assert "tick" in snapshot
        assert "alive_neurons" in snapshot
        assert snapshot["tick"] == tick

    # 5. Verify Output
    final_metrics = observer.get_tick_metrics(total_ticks)
    logger.log_summary(total_ticks, final_metrics)
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    
    # Read back the log file to verify content
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # Should have at least one line per tick + summary
        assert len(lines) >= total_ticks + 1 
        
        # Verify JSON validity of the first line
        first_record = json.loads(lines[0])
        assert "metrics" in first_record
        assert first_record["metrics"]["tick"] == 1

    # 6. Test Learning Module (Symbolic)
    bank = MemoryBank()
    bank.add_many(["1+1=2", "1+2=3", "2+2=4"])
    miner = PatternMiner()
    miner.fit(bank.expressions)
    engine = QueryEngine(miner=miner)

    # Verify learning capabilities
    writer = JsonlWriter(output_path)
    queries = ["1+2=?", "2+2=?", "3+3=?"]
    results = {}
    
    known = 0
    for query in queries:
        answer = engine.query(query)
        results[query] = answer
        if answer != "unknown":
            known += 1
        writer.append({"type": "learning", "query": query, "answer": answer})
    
    # Assertions for learning correctness
    assert results["1+2=?"] == "3"  # Should be learned
    assert results["2+2=?"] == "4"  # Should be learned
    assert results["3+3=?"] == "unknown" # Should not be learned
    
    writer.append(
        {
            "type": "learning_summary",
            "training_samples": len(bank.expressions),
            "queries": len(queries),
            "known": known,
        }
    )
    
    assert known == 2
