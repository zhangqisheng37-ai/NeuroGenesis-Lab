"""共享测试夹具 — 提供标准化的测试对象构造。

提供 4 个核心 fixture：
- sample_config: 标准 SimConfig 实例
- small_network: 10 个 LIF 神经元环形连接的 Network
- small_vectorized: 100 个神经元已初始化的 VectorizedNetwork
- tmp_output_dir: 临时输出目录
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neurogenesis_lab.config import SimConfig
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron
from neurogenesis_lab.vectorized import VectorizedNetwork


def _build_ring_network(
    n: int,
    *,
    max_neurons: int,
    energy_cap: float,
    min_neurons: int,
    neuron_kwargs: dict[str, Any] | None = None,
    network_kwargs: dict[str, Any] | None = None,
) -> Network:
    kwargs = neuron_kwargs or {}
    extra_network_kwargs = network_kwargs or {}
    neurons = [LIFNeuron(**kwargs) for _ in range(n)]
    net = Network(
        neurons={neuron.id: neuron for neuron in neurons},
        max_neurons=max_neurons,
        energy_cap=energy_cap,
        min_neurons=min_neurons,
        **extra_network_kwargs,
    )
    for i, neuron in enumerate(neurons):
        target = neurons[(i + 1) % n]
        net.connect(neuron.id, target.id, weight=1.0, delay=1)
    return net


@pytest.fixture
def ring_network_factory() -> Any:
    """返回可配置的环形 Network 构造器。"""

    def _factory(
        n: int = 10,
        *,
        max_neurons: int | None = None,
        energy_cap: float = 5.0,
        min_neurons: int = 2,
        neuron_kwargs: dict[str, Any] | None = None,
        network_kwargs: dict[str, Any] | None = None,
    ) -> Network:
        resolved_max = max_neurons if max_neurons is not None else max(n * 2, 50)
        return _build_ring_network(
            n,
            max_neurons=resolved_max,
            energy_cap=energy_cap,
            min_neurons=min_neurons,
            neuron_kwargs=neuron_kwargs,
            network_kwargs=network_kwargs,
        )

    return _factory


@pytest.fixture
def sample_config() -> SimConfig:
    """返回标准 SimConfig 对象（默认参数）。"""
    return SimConfig()


@pytest.fixture
def small_network(ring_network_factory: Any) -> Network:
    """返回 10 个 LIF 神经元的已连接 Network（环形拓扑）。"""
    return ring_network_factory(n=10, max_neurons=50, energy_cap=5.0, min_neurons=2)


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
