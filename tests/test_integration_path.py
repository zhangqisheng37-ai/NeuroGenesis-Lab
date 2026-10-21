"""教学路径与端到端集成测试（原 01~05 脚本与 test_06_integration 合并）。"""

import json

from neurogenesis_lab.encoding import Encoder, Event, Tokenizer
from neurogenesis_lab.learning import (
    MemoryBank,
    PatternMiner,
    QueryEngine,
    RuleComposer,
    RuleDiscovery,
)
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import AdExNeuron, IzhikevichNeuron, LIFNeuron
from neurogenesis_lab.observability import JsonlWriter, NetworkObserver, ObservabilityLogger


def test_encode_expr_and_event_order() -> None:
    encoder = Encoder()
    expr = "1+1=2"

    tokens = encoder.encode_expr(expr)
    events = encoder.to_events(tokens, src_id="test-src")

    assert tokens == [ord(symbol) for symbol in expr]
    assert len(events) == len(tokens)
    assert [event.t for event in events] == list(range(1, len(expr) + 1))
    assert [event.src_id for event in events] == ["test-src"] * len(events)
    assert [event.meta["symbol"] for event in events] == list(expr)


# ======================================================================
# 合并自 02_network_spread.py：02_network_spread.py
# ======================================================================


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


# ======================================================================
# 合并自 03_learn_table.py：03_learn_table.py
# ======================================================================


def test_pattern_table_answers_known_queries() -> None:
    bank = MemoryBank()
    bank.add_many(["1+1=2", "1+2=3"])

    miner = PatternMiner()
    miner.fit(bank.expressions)
    engine = QueryEngine(miner=miner)

    assert engine.query("1+1=?") == "2"
    assert engine.query("1+2=?") == "3"
    assert engine.query("1+3=?") == "unknown"


# ======================================================================
# 合并自 04_rule_discovery.py：04_rule_discovery.py
# ======================================================================


def test_rule_discovery_and_composer_fill_missing_token() -> None:
    tokenizer = Tokenizer()
    samples = [
        tokenizer.encode("1+1=2"),
        tokenizer.encode("1+2=3"),
        tokenizer.encode("2+1=3"),
    ]

    store = RuleDiscovery(min_support=2).discover(samples)
    composer = RuleComposer(store=store)

    query = tokenizer.encode("1+3=?")
    missing_token = tokenizer.encode("?")[0]
    masked = [token if token != missing_token else -1 for token in query]
    filled = composer.apply(masked, missing_token=-1)
    answer = "".join(tokenizer.decode_token(token) for token in filled)

    assert len(store.bigram_support) > 0
    assert len(store.trigram_support) > 0
    assert answer == "1+3=4"


# ======================================================================
# 合并自 05_evolution.py：05_evolution.py
# ======================================================================


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
    network = Network(neurons={doomed.id: doomed}, max_neurons=8, min_neurons=0)

    network.tick(1)

    assert len(network.neurons) == 0
    assert len(network.corpses) == 1


# ======================================================================
# 合并自 test_06_integration.py：test_06_integration.py
# ======================================================================


# Ensure src is in path for imports to work without package installation


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
