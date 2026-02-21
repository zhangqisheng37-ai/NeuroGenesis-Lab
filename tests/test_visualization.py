"""可视化模块测试"""

import pytest
from pathlib import Path
from neurogenesis_lab.visualization import (
    check_matplotlib,
    load_jsonl_data,
    VisualizationError,
    MATPLOTLIB_AVAILABLE,
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
        {"metrics": {"tick": i, "mean_energy": 1.0 + i * 0.1}}
        for i in range(1, 51)
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
            "learning": {"total_spikes": i * 5, "total_propagations": i * 3}
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
        (t, f"n{i}") 
        for t in range(1, 101) 
        for i in range(10) 
        if (t + i) % 7 == 0
    ]
    
    output_path = tmp_path / "raster.png"
    fig = plot_spike_raster(spike_data, output_path=output_path, show=False, max_neurons=10)
    
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
        positions, 
        connections, 
        energies=energies,
        output_path=output_path, 
        show=False
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
                "prediction_error": 1.0 / (i + 1)
            },
            "learning": {
                "total_spikes": i * 5,
                "total_propagations": i * 3,
                "pe_moving_avg": 0.5
            }
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


def test_visualization_error_on_empty_spike_data():
    """测试空脉冲数据时抛出错误"""
    if not MATPLOTLIB_AVAILABLE:
        pytest.skip("matplotlib 未安装")
    
    from neurogenesis_lab.visualization import plot_spike_raster
    
    with pytest.raises(VisualizationError, match="没有脉冲数据"):
        plot_spike_raster([], show=False)

