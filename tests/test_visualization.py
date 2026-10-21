"""可视化模块测试"""

import pytest
import matplotlib

matplotlib.use("Agg")
from neurogenesis_lab.visualization import (
    check_matplotlib,
    load_jsonl_data,
    VisualizationError,
    MATPLOTLIB_AVAILABLE,
    LiveMonitor,
)


def test_matplotlib_check():
    """测试 matplotlib 可用性检查"""
    if MATPLOTLIB_AVAILABLE:
        # 应该不抛出异常
        check_matplotlib()
    else:
        with pytest.raises(VisualizationError, match="matplotlib 未安装"):
            check_matplotlib()


def test_load_jsonl_data(tmp_path):
    """测试 JSONL 数据加载"""
    import json

    # 创建测试数据
    test_data = [
        {"tick": 1, "value": 10},
        {"tick": 2, "value": 20},
        {"tick": 3, "value": 30},
    ]

    # 写入文件
    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # 加载并验证
    loaded = load_jsonl_data(jsonl_path)
    assert len(loaded) == 3
    assert loaded[0]["tick"] == 1
    assert loaded[1]["value"] == 20


def test_load_jsonl_gz_data(tmp_path):
    """测试压缩 JSONL 数据加载"""
    import json
    import gzip

    test_data = [
        {"tick": 1, "metrics": {"alive": 10}},
        {"tick": 2, "metrics": {"alive": 12}},
    ]

    # 写入压缩文件
    gz_path = tmp_path / "test.jsonl.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    # 加载并验证
    loaded = load_jsonl_data(gz_path)
    assert len(loaded) == 2
    assert loaded[0]["metrics"]["alive"] == 10


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_plot_population_curve(tmp_path):
    """测试种群曲线绘制"""
    from neurogenesis_lab.visualization import plot_population_curve

    # 准备测试数据
    data = [
        {"metrics": {"tick": i, "alive_neurons": 10 + i, "total_neurons": 15 + i}}
        for i in range(1, 11)
    ]

    output_path = tmp_path / "population.png"

    # 绘制（不显示）
    fig = plot_population_curve(data, output_path=output_path, show=False)

    assert output_path.exists()
    assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_plot_energy_distribution(tmp_path):
    """测试能量分布绘制"""
    from neurogenesis_lab.visualization import plot_energy_distribution

    data = [
        {"metrics": {"tick": i, "mean_energy": 1.0 + i * 0.1}} for i in range(1, 51)
    ]

    output_path = tmp_path / "energy.png"
    fig = plot_energy_distribution(data, output_path=output_path, show=False)

    assert output_path.exists()
    assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_plot_learning_metrics(tmp_path):
    """测试学习指标绘制"""
    from neurogenesis_lab.visualization import plot_learning_metrics

    data = [
        {
            "metrics": {"tick": i, "prediction_error": 1.0 / (i + 1)},
            "learning": {"total_spikes": i * 5, "total_propagations": i * 3},
        }
        for i in range(1, 21)
    ]

    output_path = tmp_path / "learning.png"
    fig = plot_learning_metrics(data, output_path=output_path, show=False)

    assert output_path.exists()
    assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_plot_spike_raster(tmp_path):
    """测试脉冲栅格图绘制"""
    from neurogenesis_lab.visualization import plot_spike_raster

    # 生成测试脉冲数据
    spike_data = [
        (t, f"n{i}") for t in range(1, 101) for i in range(10) if (t + i) % 7 == 0
    ]

    output_path = tmp_path / "raster.png"
    fig = plot_spike_raster(
        spike_data, output_path=output_path, show=False, max_neurons=10
    )

    assert output_path.exists()
    assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_plot_network_topology(tmp_path):
    """测试网络拓扑绘制"""
    from neurogenesis_lab.visualization import plot_network_topology

    positions = {f"n{i}": (float(i % 5), float(i // 5)) for i in range(10)}
    connections = [(f"n{i}", f"n{(i+1)%10}") for i in range(10)]
    energies = {f"n{i}": 1.0 + i * 0.1 for i in range(10)}

    output_path = tmp_path / "topology.png"
    fig = plot_network_topology(
        positions, connections, energies=energies, output_path=output_path, show=False
    )

    assert output_path.exists()
    assert fig is not None


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_generate_report(tmp_path):
    """测试完整报告生成"""
    import json
    from neurogenesis_lab.visualization import generate_report

    # 创建测试数据文件
    data = [
        {
            "metrics": {
                "tick": i,
                "alive_neurons": 10 + i,
                "total_neurons": 15 + i,
                "mean_energy": 1.5 - i * 0.01,
                "prediction_error": 1.0 / (i + 1),
            },
            "learning": {
                "total_spikes": i * 5,
                "total_propagations": i * 3,
                "pe_moving_avg": 0.5,
            },
        }
        for i in range(1, 51)
    ]

    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    output_dir = tmp_path / "report"

    # 生成报告
    generate_report(jsonl_path, output_dir, show=False)

    # 验证输出文件
    assert (output_dir / "population.png").exists()
    assert (output_dir / "energy.png").exists()
    assert (output_dir / "learning.png").exists()


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_live_monitor_on_tick_updates_network_path(ring_network_factory) -> None:
    monitor = LiveMonitor(plot_interval=1, max_points=10)
    net = ring_network_factory(n=2, min_neurons=0)
    neurons = list(net.neurons.values())
    net.connect(neurons[0].id, neurons[1].id, weight=1.0, delay=1)

    snapshot = {
        "alive_neurons": 2,
        "total_neurons": 2,
        "mean_energy": 1.0,
        "prediction_error": 0.1,
        "rule_count": 0,
    }
    monitor.on_tick(1, snapshot, net)

    assert monitor._initialized is True
    assert monitor._ticks[-1] == 1
    assert monitor._alive[-1] == 2


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_live_monitor_on_tick_vectorized_path() -> None:
    from neurogenesis_lab.vectorized import VectorizedNetwork

    monitor = LiveMonitor(plot_interval=1, max_points=10)
    vnet = VectorizedNetwork()
    vnet.init_neurons(5)

    snapshot = {
        "alive_neurons": 5,
        "total_neurons": 5,
        "mean_energy": 1.0,
        "prediction_error": 0.2,
        "rule_count": 1,
    }
    monitor.on_tick(1, snapshot, vnet)
    assert monitor._initialized is True
    assert monitor._ticks[-1] == 1


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_live_monitor_on_tick_plot_exception_is_swallowed(monkeypatch) -> None:
    monitor = LiveMonitor(plot_interval=1)

    def _raise() -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(monitor, "_update_plots", _raise)
    monitor.on_tick(
        1,
        {
            "alive_neurons": 1,
            "total_neurons": 1,
            "mean_energy": 1.0,
            "prediction_error": 0.0,
            "rule_count": 0,
        },
        None,
    )
    assert monitor._ticks[-1] == 1


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib 未安装")
def test_live_monitor_finalize_exception_is_swallowed(monkeypatch) -> None:
    monitor = LiveMonitor(plot_interval=1)
    monitor._init_figure()

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monitor.finalize()
