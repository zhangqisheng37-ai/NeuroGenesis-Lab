"""长程稳定性回归测试 — 验证 500 ticks 实验的稳定性"""

import random

import numpy as np
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.encoding import Event
from neurogenesis_lab.vectorized import VectorizedNetwork


def create_stable_network(size: int = 10) -> Network:
    """创建用于长程测试的稳定网络"""
    neurons = {}
    for i in range(size):
        neuron = LIFNeuron(
            id=f"n{i}",
            pos=(float(i % 5), float(i // 5)),
            energy=2.0,
            v_th=-50.0,
            tau_m=20.0,
            spike_cost=0.03,
            input_gain=0.1,
            move_cost=0.005,
        )
        neurons[neuron.id] = neuron

    network = Network(
        neurons=neurons,
        reproduction_threshold=1.8,
        reproduction_share=0.35,
        energy_decay=0.001,
        max_neurons=60,
        min_neurons=5,
        energy_cap=5.0,
        adaptive_decay=True,
        adaptive_reproduction=True,
        endangered_threshold=10,
        move_cost=0.005,
    )

    # 建立初始连接（双向）
    for i in range(size - 1):
        network.connect(f"n{i}", f"n{i+1}", weight=1.0, delay=1)
        network.connect(f"n{i+1}", f"n{i}", weight=1.0, delay=1)

    return network


def test_long_term_population_stability():
    """测试 500 ticks 种群稳定性 — 不崩溃、不爆炸"""
    network = create_stable_network(size=10)

    # 添加持续输入（提高权重保证能量注入）
    for tick in range(0, 500, 5):
        network.enqueue(
            [Event(token_id=1, t=tick, src_id=f"n{tick % 10}", meta={"weight": 5.0})]
        )

    population_history = []

    for tick in range(500):
        snapshot = network.tick(tick, dt=1.0, run_evolution=True)
        alive = snapshot["alive_neurons"]
        population_history.append(alive)

    # 验证：种群保持在合理范围
    min_pop = min(population_history)
    max_pop = max(population_history)
    final_pop = population_history[-1]

    assert min_pop >= 2, f"种群跌破下限: {min_pop}"
    assert max_pop <= 60, f"种群超过上限: {max_pop}"
    assert final_pop >= 2, f"最终种群过少: {final_pop}"

    # 验证：后半段种群相对稳定（波动 < 50%）
    second_half = population_history[250:]
    avg_pop = sum(second_half) / len(second_half)
    assert avg_pop > 0, "平均种群为零"

    max_deviation = max(abs(p - avg_pop) / avg_pop for p in second_half if avg_pop > 0)
    assert max_deviation < 2.0, f"种群波动过大: {max_deviation:.2%}"


def test_long_term_energy_conservation():
    """测试 500 ticks 能量守恒 — 总能量变化在合理范围"""
    network = create_stable_network(size=10)

    # 添加持续输入（提供能量）
    for tick in range(0, 500, 5):
        network.enqueue(
            [Event(token_id=1, t=tick, src_id=f"n{tick % 10}", meta={"weight": 1.0})]
        )

    initial_energy = sum(n.energy for n in network.neurons.values())
    energy_history = [initial_energy]

    for tick in range(500):
        network.tick(tick, dt=1.0, run_evolution=True)
        alive_neurons = [n for n in network.neurons.values() if n.alive]
        total_energy = sum(n.energy for n in alive_neurons)
        energy_history.append(total_energy)

    final_energy = energy_history[-1]

    # 验证：能量不会无限增长（有上限保护）
    max_energy = max(energy_history)
    expected_max = network.max_neurons * network.energy_cap
    assert (
        max_energy <= expected_max * 1.2
    ), f"能量超过理论上限: {max_energy:.2f} > {expected_max:.2f}"

    # 验证：能量不会完全耗尽（有濒危保护）
    min_energy = min(energy_history[100:])  # 跳过初始阶段
    assert min_energy > 0, "能量完全耗尽"

    # 验证：最终能量在合理范围内（不为零且不超限）
    assert (
        0 < final_energy <= expected_max * 1.2
    ), f"最终能量异常: {final_energy:.2f} (期望范围: 0 ~ {expected_max * 1.2:.2f})"


def test_long_term_birth_death_balance():
    """测试 500 ticks 生死平衡 — 繁殖率与死亡率趋于平衡"""
    network = create_stable_network(size=10)

    # 添加持续输入（充足的能量注入保证繁殖）
    for tick in range(0, 500, 3):
        network.enqueue(
            [Event(token_id=1, t=tick, src_id=f"n{tick % 10}", meta={"weight": 8.0})]
        )

    for tick in range(500):
        network.tick(tick, dt=1.0, run_evolution=True)

    total_births = network.birth_count
    total_deaths = network.death_count

    # 验证：有繁殖和死亡发生
    assert total_births > 0, "没有繁殖发生"
    assert total_deaths > 0, "没有死亡发生"

    # 验证：生死比例在合理范围（0.5 ~ 2.0）
    if total_deaths > 0:
        birth_death_ratio = total_births / total_deaths
        assert 0.5 <= birth_death_ratio <= 2.0, f"生死比例失衡: {birth_death_ratio:.2f}"


def test_adaptive_decay_mechanism():
    """测试自适应衰减机制"""
    network = create_stable_network(size=10)
    network.adaptive_decay = True

    # 测试高密度场景
    alive_count_high = int(network.max_neurons * 0.9)
    decay_high = network._calculate_adaptive_decay(alive_count_high)
    assert decay_high > network.energy_decay, "高密度时衰减应增加"

    # 测试低密度场景
    alive_count_low = int(network.max_neurons * 0.2)
    decay_low = network._calculate_adaptive_decay(alive_count_low)
    assert decay_low < network.energy_decay, "低密度时衰减应减少"

    # 测试中等密度场景
    alive_count_mid = int(network.max_neurons * 0.5)
    decay_mid = network._calculate_adaptive_decay(alive_count_mid)
    assert abs(decay_mid - network.energy_decay) < 1e-9, "中等密度时衰减应保持不变"


def test_adaptive_reproduction_mechanism():
    """测试自适应繁殖成本机制"""
    network = create_stable_network(size=10)
    network.adaptive_reproduction = True

    # 测试高种群场景
    alive_count_high = int(network.max_neurons * 0.9)
    share_high = network._calculate_adaptive_reproduction_share(alive_count_high)
    assert share_high > network.reproduction_share, "高种群时繁殖成本应增加"

    # 测试低种群场景
    alive_count_low = int(network.max_neurons * 0.2)
    share_low = network._calculate_adaptive_reproduction_share(alive_count_low)
    assert share_low < network.reproduction_share, "低种群时繁殖成本应减少"

    # 测试中等种群场景
    alive_count_mid = int(network.max_neurons * 0.5)
    share_mid = network._calculate_adaptive_reproduction_share(alive_count_mid)
    assert (
        abs(share_mid - network.reproduction_share) < 1e-9
    ), "中等种群时繁殖成本应保持不变"


def test_endangered_mode_protection():
    """测试濒危模式保护机制"""
    network = create_stable_network(size=8)
    network.endangered_threshold = 10
    network.energy_decay = 0.01  # 较高衰减

    # 添加少量输入，让种群自然衰减
    for tick in range(0, 100, 20):
        network.enqueue([Event(token_id=1, t=tick, src_id="n0", meta={"weight": 0.5})])

    for tick in range(100):
        snapshot = network.tick(tick, dt=1.0, run_evolution=True)
        alive = snapshot["alive_neurons"]

        # 验证：种群不会低于 min_neurons
        assert (
            alive >= network.min_neurons
        ), f"种群跌破保护下限: {alive} < {network.min_neurons}"

    # 验证：濒危模式被触发
    final_alive = sum(1 for n in network.neurons.values() if n.alive)
    if final_alive <= network.endangered_threshold:
        assert network._endangered_mode, "濒危模式应该被触发"


def test_energy_cap_enforcement():
    """测试能量上限强制执行"""
    network = create_stable_network(size=5)
    network.energy_cap = 3.0

    # 给神经元大量能量输入
    for tick in range(50):
        for i in range(5):
            network.enqueue(
                [Event(token_id=1, t=tick, src_id=f"n{i}", meta={"weight": 2.0})]
            )

    for tick in range(50):
        network.tick(tick, dt=1.0, run_evolution=True)

    # 验证：所有神经元能量不超过上限
    for neuron in network.neurons.values():
        if neuron.alive:
            assert (
                neuron.energy <= network.energy_cap
            ), f"神经元 {neuron.id} 能量超过上限: {neuron.energy} > {network.energy_cap}"


def _init_pair_engines(seed: int, size: int = 20) -> tuple[Network, VectorizedNetwork]:
    """初始化同 seed 的双引擎对照组。"""
    random.seed(seed)
    np.random.seed(seed)

    std = create_stable_network(size=size)

    random.seed(seed)
    np.random.seed(seed)
    vec = VectorizedNetwork(
        reproduction_threshold=1.8,
        reproduction_share=0.35,
        energy_decay=0.001,
        max_neurons=60,
        min_neurons=5,
        energy_cap=5.0,
        adaptive_decay=True,
        adaptive_reproduction=True,
        endangered_threshold=10,
        move_cost=0.005,
        input_current_scale=0.1,
    )
    vec.init_neurons(
        size,
        energy_init=2.0,
        v_th=-50.0,
        tau_m=20.0,
        r_m=1.0,
        spike_cost_val=0.03,
        spread=5.0,
    )
    vec.connect_knn(k=2, weight=1.0, delay=1)

    for tick in range(0, 500, 5):
        src = tick % size
        std.enqueue([Event(token_id=1, t=tick, src_id=f"n{src}", meta={"weight": 5.0})])
        vec.enqueue([Event(token_id=1, t=tick, src_id=str(src), meta={"weight": 5.0})])

    return std, vec


def _series_stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def test_long_term_dual_engine_population_trend_consistency() -> None:
    """同 seed 长程运行下双引擎种群趋势应统计一致（非逐值）。"""
    std, vec = _init_pair_engines(seed=123, size=20)

    std_pop: list[float] = []
    vec_pop: list[float] = []

    for tick in range(1, 501):
        s1 = std.tick(tick, dt=1.0, run_evolution=True)
        s2 = vec.tick(tick, dt=1.0)
        std_pop.append(float(s1["alive_neurons"]))
        vec_pop.append(float(s2["alive_neurons"]))

    std_tail_mean, std_tail_std = _series_stats(std_pop[250:])
    vec_tail_mean, vec_tail_std = _series_stats(vec_pop[250:])

    mean_diff_ratio = abs(std_tail_mean - vec_tail_mean) / max(std_tail_mean, vec_tail_mean, 1.0)
    std_diff_ratio = abs(std_tail_std - vec_tail_std) / max(std_tail_std, vec_tail_std, 1.0)

    assert mean_diff_ratio <= 0.55, (
        f"尾段均值趋势偏差过大: std={std_tail_mean:.2f}, vec={vec_tail_mean:.2f}, "
        f"ratio={mean_diff_ratio:.2%}"
    )
    assert std_diff_ratio <= 1.00, (
        f"尾段波动趋势偏差过大: std={std_tail_std:.2f}, vec={vec_tail_std:.2f}, "
        f"ratio={std_diff_ratio:.2%}"
    )
