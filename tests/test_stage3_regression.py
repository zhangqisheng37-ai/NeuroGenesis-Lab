"""阶段三回归测试 — 动力学激活与引擎可比性"""

import random

import numpy as np

from neurogenesis_lab.config import SimConfig, seed_all
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import AdExNeuron, IzhikevichNeuron, LIFNeuron
from neurogenesis_lab.vectorized import VectorizedNetwork


def test_default_like_run_has_nonzero_spike_and_propagation() -> None:
    """默认 run 等价配置在 100 tick 内应有非零放电与传播。"""
    cfg = SimConfig()
    seed_all(cfg.seed)

    neuron_types = [LIFNeuron, IzhikevichNeuron, AdExNeuron]
    common_neuron_kwargs = {
        "v_rest": cfg.neuron.v_rest,
        "v_reset": cfg.neuron.v_reset,
        "v_th": cfg.neuron.v_th,
        "tau_m": cfg.neuron.tau_m,
        "r_m": cfg.neuron.r_m,
        "spike_cost": cfg.neuron.spike_cost,
        "input_gain": cfg.neuron.input_gain,
        "mem_size": cfg.neuron.mem_size,
        "move_cost": cfg.network.move_cost,
    }
    neurons = [
        neuron_types[i % len(neuron_types)](**common_neuron_kwargs) for i in range(8)
    ]
    net = Network(
        neurons={n.id: n for n in neurons},
        reproduction_threshold=cfg.network.reproduction_threshold,
        reproduction_share=cfg.network.reproduction_share,
        mutation_scale=cfg.network.mutation_scale,
        energy_decay=cfg.network.energy_decay,
        max_neurons=cfg.network.max_neurons,
        max_links=cfg.network.max_links,
        corpse_ttl=cfg.network.corpse_ttl,
        scavenger_radius=cfg.network.scavenger_radius,
        input_current_scale=cfg.network.input_current_scale,
        move_cost=cfg.network.move_cost,
        fitness_weights=cfg.evolution.fitness_weights,
        congestion_penalty=cfg.evolution.congestion_penalty,
        min_neurons=cfg.evolution.min_neurons,
    )

    for i, neuron in enumerate(neurons):
        target = neurons[(i + 1) % len(neurons)]
        net.connect(neuron.id, target.id, weight=1.0, delay=1)

    encoder = Encoder()
    tokens = encoder.encode_expr("1+2=3")
    for tick_start in range(1, cfg.ticks + 1, 5):
        base_events = encoder.to_events(tokens, src_id=neurons[0].id)
        net.enqueue(
            [
                Event(
                    token_id=e.token_id,
                    t=e.t + tick_start,
                    src_id=e.src_id,
                    meta={**e.meta, "weight": 40.0},
                )
                for e in base_events
                if e.t + tick_start <= cfg.ticks
            ]
        )

    for tick in range(1, cfg.ticks + 1):
        net.tick(tick, dt=cfg.dt)

    summary = net.get_summary()
    assert (
        summary["total_spikes"] > 0
    ), f"默认配置应触发放电，实际 total_spikes={summary['total_spikes']}"
    assert (
        summary["total_propagations"] > 0
    ), f"默认配置应触发传播，实际 total_propagations={summary['total_propagations']}"


def test_vectorized_and_original_lif_are_statistically_comparable() -> None:
    """统一 LIF 条件下，两引擎的放电统计应在合理误差内。"""
    random.seed(7)
    np.random.seed(7)

    n = 16
    ticks = 30

    lif_kwargs = {
        "v_rest": -65.0,
        "v_reset": -65.0,
        "v_th": -64.0,
        "tau_m": 1.0,
        "r_m": 30.0,
        "move_cost": 0.0,
    }

    neurons = [LIFNeuron(**lif_kwargs) for _ in range(n)]
    net = Network(
        neurons={item.id: item for item in neurons},
        energy_decay=0.0,
        move_cost=0.0,
        input_current_scale=5.0,
        reproduction_threshold=999.0,
        max_neurons=n,
        min_neurons=0,
    )

    vnet = VectorizedNetwork(
        energy_decay=0.0,
        move_cost=0.0,
        input_current_scale=5.0,
        reproduction_threshold=999.0,
        max_neurons=n,
        min_neurons=0,
    )
    vnet.init_neurons(
        n,
        v_rest=-65.0,
        v_reset=-65.0,
        v_th=-64.0,
        tau_m=1.0,
        r_m=30.0,
        energy_init=2.0,
    )

    for tick in range(1, ticks + 1):
        for item in neurons:
            net.enqueue(
                [Event(token_id=1, t=tick, src_id=item.id, meta={"weight": 1.0})]
            )
        net.tick(tick, dt=1.0)
        vnet.tick(tick, dt=1.0, input_currents=np.ones(n, dtype=np.float64))

    original_summary = net.get_summary()
    vector_summary = vnet.get_summary()

    assert (
        original_summary["alive_neurons"] == vector_summary["alive_neurons"]
    ), "统一无衰减参数下，存活数量应一致"

    original_spikes = original_summary["total_spikes"]
    vector_spikes = vector_summary["total_spikes"]
    diff_ratio = abs(original_spikes - vector_spikes) / max(1, original_spikes)
    assert diff_ratio <= 0.2, (
        "两引擎总放电应统计可比: "
        f"original={original_spikes}, vectorized={vector_spikes}, diff_ratio={diff_ratio:.3f}"
    )


def test_vectorized_and_original_delay_routing_are_behaviorally_consistent() -> None:
    """延迟路由行为应一致：传播后在 delay tick 出现就绪输入。"""
    random.seed(11)
    np.random.seed(11)

    a = LIFNeuron(
        v_rest=-65.0, v_reset=-65.0, v_th=-64.0, tau_m=1.0, r_m=30.0, move_cost=0.0
    )
    b = LIFNeuron(
        v_rest=-65.0, v_reset=-65.0, v_th=-64.0, tau_m=1.0, r_m=30.0, move_cost=0.0
    )
    net = Network(
        neurons={a.id: a, b.id: b},
        energy_decay=0.0,
        move_cost=0.0,
        input_current_scale=5.0,
        reproduction_threshold=999.0,
        min_neurons=0,
        max_neurons=2,
    )
    net.connect(a.id, b.id, weight=1.0, delay=2)
    net.enqueue([Event(token_id=1, t=1, src_id=a.id, meta={"weight": 1.0})])

    vnet = VectorizedNetwork(
        energy_decay=0.0,
        move_cost=0.0,
        input_current_scale=5.0,
        reproduction_threshold=999.0,
        max_neurons=2,
        min_neurons=0,
    )
    vnet.init_neurons(
        2,
        v_rest=-65.0,
        v_reset=-65.0,
        v_th=-64.0,
        tau_m=1.0,
        r_m=30.0,
        energy_init=2.0,
    )
    vnet.connect_knn(k=1, weight=1.0, delay=2)
    vnet.enqueue([Event(token_id=1, t=1, src_id=0, meta={"weight": 1.0})])

    s1 = net.tick(1)
    vs1 = vnet.tick(1)
    assert s1["propagation_count"] > 0
    assert vs1["propagation_count"] > 0

    s2 = net.tick(2)
    vs2 = vnet.tick(2)
    assert s2.get("ready_events", 0) == 0
    assert vs2.get("ready_events", 0) == 0

    s3 = net.tick(3)
    vs3 = vnet.tick(3)
    assert s3.get("ready_events", 0) > 0
    assert vs3.get("ready_events", 0) > 0
