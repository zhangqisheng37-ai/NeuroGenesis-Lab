# visualization — 可视化系统

实时监控面板与离线图表生成（需要 `matplotlib>=3.5`）。

## 核心类

- **`LiveMonitor`** — 实时监控面板，在仿真运行时动态更新图表
- **`VisualizationError`** — 可视化异常（matplotlib 未安装时抛出）

## 绘图函数

- `plot_population_curve(data, output_path, show)` — 种群数量曲线
- `plot_energy_distribution(data, output_path, show)` — 能量分布直方图
- `plot_learning_metrics(data, output_path, show)` — 学习指标趋势
- `plot_spike_raster(spike_data, output_path, show)` — 放电栅格图
- `plot_network_topology(positions, connections, energies, ...)` — 网络拓扑可视化

## 工具函数

- `check_matplotlib()` — 检查 matplotlib 可用性
- `load_jsonl_data(path)` — 从 JSONL 文件加载数据
- `generate_report(data_path, output_dir, show)` — 生成完整离线报告

::: neurogenesis_lab.visualization
