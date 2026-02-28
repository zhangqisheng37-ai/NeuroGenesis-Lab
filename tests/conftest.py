"""共享测试夹具 — 提供标准化的测试对象构造。

提供 4 个核心 fixture：
- sample_config: 标准 SimConfig 实例
- small_network: 10 个 LIF 神经元环形连接的 Network
- small_vectorized: 100 个神经元已初始化的 VectorizedNetwork
- tmp_output_dir: 临时输出目录
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurogenesis_lab.config import SimConfig
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.vectorized import VectorizedNetwork


@pytest.fixture
def sample_config() -> SimConfig:
    """返回标准 SimConfig 对象（默认参数）。"""
    return SimConfig()


@pytest.fixture
def small_network() -> Network:
    """返回 10 个 LIF 神经元的已连接 Network（环形拓扑）。"""
    neurons = [LIFNeuron() for _ in range(10)]
    net = Network(
        neurons={n.id: n for n in neurons},
        max_neurons=50,
        energy_cap=5.0,
        min_neurons=2,
    )
    # 环形连接
    for i, neuron in enumerate(neurons):
        target = neurons[(i + 1) % len(neurons)]
        net.connect(neuron.id, target.id, weight=1.0, delay=1)
    return net


@pytest.fixture
def small_vectorized() -> VectorizedNetwork:
    """返回 100 个神经元的已初始化 VectorizedNetwork。"""
    vnet = VectorizedNetwork(max_neurons=200, min_neurons=2)
    vnet.init_neurons(100, energy_init=1.5)
    vnet.connect_knn(k=3)
    return vnet


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """返回一个临时输出目录。"""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sensitive_neurons() -> list[LIFNeuron]:
    """返回高敏感配置的 LIF 神经元列表（易放电）。"""
    return [LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0) for _ in range(5)]
