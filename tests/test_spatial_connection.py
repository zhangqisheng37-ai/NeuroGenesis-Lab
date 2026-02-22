import pytest
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron


def test_spatial_connection():
    """测试基于空间距离的连接策略"""
    network = Network(neurons={})

    # 创建 3 个神经元，位置不同
    n1 = LIFNeuron(id="n1", pos=(0.0, 0.0))
    n2 = LIFNeuron(id="n2", pos=(1.0, 0.0))  # 距离 n1 为 1.0
    n3 = LIFNeuron(id="n3", pos=(10.0, 0.0))  # 距离 n1 为 10.0

    network.add_neuron(n1)
    network.add_neuron(n2)
    network.add_neuron(n3)

    # 使用半径 5.0 进行连接
    network.connect_by_proximity(radius=5.0, max_links_per_neuron=5)

    # n1 应该连接到 n2，但不连接到 n3
    assert "n2" in n1.links
    assert "n3" not in n1.links

    # 检查权重和延迟
    link = n1.links["n2"]
    assert link["weight"] > 0.5  # 距离近，权重高
    assert link["delay"] >= 1  # 延迟至少为 1


def test_spatial_evolution():
    """测试繁殖时的空间位置继承"""
    network = Network(neurons={})
    parent = LIFNeuron(id="parent", pos=(10.0, 10.0), energy=5.0)
    network.add_neuron(parent)

    # 触发繁殖
    network.check_reproduction()

    # 应该有一个子代
    assert len(network.neurons) == 2

    # 找到子代
    child = next(n for n in network.neurons.values() if n.id != "parent")

    # 子代位置应该在母体附近（高斯扰动）
    dx = abs(child.pos[0] - parent.pos[0])
    dy = abs(child.pos[1] - parent.pos[1])

    assert dx <= 1.0
    assert dy <= 1.0


def test_neuron_movement():
    """测试神经元移动消耗能量"""
    n = LIFNeuron(id="n1", pos=(0.0, 0.0), energy=1.0, move_cost=0.1)

    # 能量充足，应该移动并消耗能量
    n.move()
    assert n.energy == 0.9
    assert n.pos != (0.0, 0.0)

    # 能量不足，不应该移动
    n.energy = 0.1
    old_pos = n.pos
    n.move()
    assert n.energy == 0.0
    assert n.pos == old_pos
    assert n.pos == old_pos
