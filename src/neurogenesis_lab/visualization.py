"""可视化模块 — 实时监控与离线分析"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.figure import Figure

    # 设置中文字体，解决乱码问题
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class VisualizationError(Exception):
    """可视化相关错误"""

    pass


def check_matplotlib() -> None:
    """检查 matplotlib 是否可用"""
    if not MATPLOTLIB_AVAILABLE:
        raise VisualizationError("matplotlib 未安装。请运行: pip install matplotlib")


def plot_population_curve(
    data: list[dict[str, Any]], output_path: Path | None = None, show: bool = True
) -> Figure:
    """绘制种群数量随时间变化曲线"""
    check_matplotlib()

    ticks = [d["metrics"]["tick"] for d in data if "metrics" in d]
    alive = [d["metrics"]["alive_neurons"] for d in data if "metrics" in d]
    total = [d["metrics"]["total_neurons"] for d in data if "metrics" in d]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ticks, alive, label="存活神经元", color="#2ecc71", linewidth=2)
    ax.plot(
        ticks, total, label="总神经元", color="#3498db", linewidth=2, linestyle="--"
    )
    ax.fill_between(ticks, alive, alpha=0.3, color="#2ecc71")

    ax.set_xlabel("Tick", fontsize=12)
    ax.set_ylabel("神经元数量", fontsize=12)
    ax.set_title("种群动态", fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_energy_distribution(
    data: list[dict[str, Any]], output_path: Path | None = None, show: bool = True
) -> Figure:
    """绘制能量分布直方图"""
    check_matplotlib()

    ticks = [d["metrics"]["tick"] for d in data if "metrics" in d]
    mean_energy = [d["metrics"]["mean_energy"] for d in data if "metrics" in d]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 能量随时间变化
    ax1.plot(ticks, mean_energy, color="#e74c3c", linewidth=2)
    ax1.fill_between(ticks, mean_energy, alpha=0.3, color="#e74c3c")
    ax1.set_xlabel("Tick", fontsize=12)
    ax1.set_ylabel("平均能量", fontsize=12)
    ax1.set_title("能量趋势", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 能量分布直方图（最后一个时刻）
    if mean_energy:
        ax2.hist(
            mean_energy[-50:], bins=20, color="#e74c3c", alpha=0.7, edgecolor="black"
        )
        ax2.set_xlabel("能量值", fontsize=12)
        ax2.set_ylabel("频次", fontsize=12)
        ax2.set_title("能量分布（最近50 ticks）", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_learning_metrics(
    data: list[dict[str, Any]], output_path: Path | None = None, show: bool = True
) -> Figure:
    """绘制学习指标曲线"""
    check_matplotlib()

    ticks = [d["metrics"]["tick"] for d in data if "metrics" in d]
    pe = [d["metrics"].get("prediction_error", 0.0) for d in data if "metrics" in d]

    # 提取学习指标
    total_spikes = [d["learning"]["total_spikes"] for d in data if "learning" in d]
    total_propagations = [
        d["learning"]["total_propagations"] for d in data if "learning" in d
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 预测误差
    ax1.plot(ticks, pe, color="#9b59b6", linewidth=2)
    ax1.fill_between(ticks, pe, alpha=0.3, color="#9b59b6")
    ax1.set_xlabel("Tick", fontsize=12)
    ax1.set_ylabel("预测误差", fontsize=12)
    ax1.set_title("预测误差收敛", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 放电与传播
    ax2.plot(
        ticks[: len(total_spikes)],
        total_spikes,
        label="总放电",
        color="#e67e22",
        linewidth=2,
    )
    ax2.plot(
        ticks[: len(total_propagations)],
        total_propagations,
        label="总传播",
        color="#1abc9c",
        linewidth=2,
    )
    ax2.set_xlabel("Tick", fontsize=12)
    ax2.set_ylabel("累计数量", fontsize=12)
    ax2.set_title("放电与传播统计", fontsize=14, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_spike_raster(
    spike_data: list[tuple[int, str]],
    output_path: Path | None = None,
    show: bool = True,
    max_neurons: int = 50,
) -> Figure:
    """绘制脉冲栅格图

    Args:
        spike_data: [(tick, neuron_id), ...] 格式的脉冲记录
        output_path: 输出路径
        show: 是否显示
        max_neurons: 最多显示的神经元数量
    """
    check_matplotlib()

    if not spike_data:
        raise VisualizationError("没有脉冲数据")

    # 提取唯一神经元ID
    neuron_ids = sorted(set(nid for _, nid in spike_data))[:max_neurons]
    neuron_to_idx = {nid: idx for idx, nid in enumerate(neuron_ids)}

    # 准备绘图数据
    ticks = [t for t, nid in spike_data if nid in neuron_to_idx]
    indices = [neuron_to_idx[nid] for _, nid in spike_data if nid in neuron_to_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(ticks, indices, s=5, color="#34495e", alpha=0.6)

    ax.set_xlabel("Tick", fontsize=12)
    ax.set_ylabel("神经元索引", fontsize=12)
    ax.set_title("脉冲栅格图", fontsize=14, fontweight="bold")
    ax.set_yticks(range(len(neuron_ids)))
    ax.set_yticklabels([f"n{i}" for i in range(len(neuron_ids))], fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_network_topology(
    positions: dict[str, tuple[float, float]],
    connections: list[tuple[str, str]],
    energies: dict[str, float] | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> Figure:
    """绘制网络拓扑图

    Args:
        positions: {neuron_id: (x, y)}
        connections: [(source_id, target_id), ...]
        energies: {neuron_id: energy} 用于颜色编码
        output_path: 输出路径
        show: 是否显示
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制连接
    for src, tgt in connections:
        if src in positions and tgt in positions:
            x1, y1 = positions[src]
            x2, y2 = positions[tgt]
            ax.plot([x1, x2], [y1, y2], color="#bdc3c7", alpha=0.3, linewidth=0.5)

    # 绘制神经元
    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]

    if energies:
        colors = [energies.get(nid, 0.0) for nid in positions.keys()]
        scatter = ax.scatter(
            xs,
            ys,
            c=colors,
            cmap="YlOrRd",
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=1,
        )
        plt.colorbar(scatter, ax=ax, label="能量")
    else:
        ax.scatter(
            xs, ys, color="#3498db", s=100, alpha=0.8, edgecolors="black", linewidth=1
        )

    ax.set_xlabel("X 坐标", fontsize=12)
    ax.set_ylabel("Y 坐标", fontsize=12)
    ax.set_title("网络拓扑结构", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def load_jsonl_data(path: Path) -> list[dict[str, Any]]:
    """加载 JSONL 数据"""
    import gzip

    data = []
    open_func = gzip.open if path.suffix == ".gz" else open

    with open_func(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def generate_report(data_path: Path, output_dir: Path, show: bool = False) -> None:
    """生成完整的可视化报告

    Args:
        data_path: JSONL 数据文件路径
        output_dir: 输出目录
        show: 是否显示图表
    """
    check_matplotlib()

    print(f"加载数据: {data_path}")
    data = load_jsonl_data(data_path)

    # 过滤出有效数据
    metrics_data = [d for d in data if "metrics" in d]

    if not metrics_data:
        raise VisualizationError("没有找到有效的 metrics 数据")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("生成种群曲线...")
    plot_population_curve(metrics_data, output_dir / "population.png", show=show)

    print("生成能量分布...")
    plot_energy_distribution(metrics_data, output_dir / "energy.png", show=show)

    print("生成学习指标...")
    plot_learning_metrics(metrics_data, output_dir / "learning.png", show=show)

    print(f"✓ 报告已生成: {output_dir}")


class LiveMonitor:
    """实时监控面板 — 使用 matplotlib.animation 动态更新 4 个核心图表"""

    def __init__(self, plot_interval: int = 5, max_points: int = 500) -> None:
        """
        Args:
            plot_interval: 每 N ticks 更新一次图表
            max_points: 最多保留的数据点数量（防止内存增长）
        """
        check_matplotlib()
        self.plot_interval = max(1, plot_interval)
        self.max_points = max_points

        # 数据缓冲区
        self._ticks: list[int] = []
        self._alive: list[int] = []
        self._total: list[int] = []
        self._mean_energy: list[float] = []
        self._pe: list[float] = []
        self._births: list[int] = []
        self._deaths: list[int] = []
        self._rule_count: list[int] = []

        # 图表对象
        self._fig: Figure | None = None
        self._axes: list[Any] = []
        self._lines: dict[str, Any] = {}
        self._initialized = False

    def _init_figure(self) -> None:
        """初始化 4 子图面板"""
        plt.ion()  # 交互模式
        self._fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        self._fig.suptitle(
            "NeuroGenesis Lab — 实时监控", fontsize=14, fontweight="bold"
        )
        self._axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

        # 子图1: 种群动态
        ax = self._axes[0]
        (self._lines["alive"],) = ax.plot(
            [], [], label="存活", color="#2ecc71", linewidth=2
        )
        (self._lines["total"],) = ax.plot(
            [], [], label="总数", color="#3498db", linewidth=2, linestyle="--"
        )
        ax.set_title("种群动态")
        ax.set_xlabel("Tick")
        ax.set_ylabel("数量")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 子图2: 能量趋势
        ax = self._axes[1]
        (self._lines["energy"],) = ax.plot([], [], color="#e74c3c", linewidth=2)
        ax.set_title("平均能量")
        ax.set_xlabel("Tick")
        ax.set_ylabel("能量")
        ax.grid(True, alpha=0.3)

        # 子图3: 预测误差
        ax = self._axes[2]
        (self._lines["pe"],) = ax.plot([], [], color="#9b59b6", linewidth=2)
        ax.set_title("预测误差")
        ax.set_xlabel("Tick")
        ax.set_ylabel("PE")
        ax.grid(True, alpha=0.3)

        # 子图4: 生死率
        ax = self._axes[3]
        (self._lines["births"],) = ax.plot(
            [], [], label="出生", color="#27ae60", linewidth=2
        )
        (self._lines["deaths"],) = ax.plot(
            [], [], label="死亡", color="#c0392b", linewidth=2
        )
        ax.set_title("生死动态")
        ax.set_xlabel("Tick")
        ax.set_ylabel("数量")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._initialized = True

    def on_tick(self, tick: int, snapshot: dict[str, Any]) -> None:
        """每个 tick 后调用，按间隔更新图表

        Args:
            tick: 当前 tick
            snapshot: Network.tick() 返回的 TickSnapshot
        """
        if tick % self.plot_interval != 0:
            return

        # 追加数据
        self._ticks.append(tick)
        self._alive.append(snapshot.get("alive_neurons", 0))
        self._total.append(snapshot.get("total_neurons", 0))
        self._mean_energy.append(snapshot.get("mean_energy", 0.0))
        self._pe.append(snapshot.get("prediction_error", 0.0))
        self._births.append(snapshot.get("births", 0))
        self._deaths.append(snapshot.get("deaths", 0))
        self._rule_count.append(snapshot.get("rule_count", 0))

        # 截断过长的历史
        if len(self._ticks) > self.max_points:
            trim = len(self._ticks) - self.max_points
            self._ticks = self._ticks[trim:]
            self._alive = self._alive[trim:]
            self._total = self._total[trim:]
            self._mean_energy = self._mean_energy[trim:]
            self._pe = self._pe[trim:]
            self._births = self._births[trim:]
            self._deaths = self._deaths[trim:]
            self._rule_count = self._rule_count[trim:]

        # 延迟初始化（确保有数据后再创建图表）
        if not self._initialized:
            self._init_figure()

        self._update_plots()

    def _update_plots(self) -> None:
        """刷新所有子图数据"""
        if not self._initialized or self._fig is None:
            return

        ticks = self._ticks

        # 种群动态
        self._lines["alive"].set_data(ticks, self._alive)
        self._lines["total"].set_data(ticks, self._total)
        ax = self._axes[0]
        ax.relim()
        ax.autoscale_view()

        # 能量趋势
        self._lines["energy"].set_data(ticks, self._mean_energy)
        ax = self._axes[1]
        ax.relim()
        ax.autoscale_view()

        # 预测误差
        self._lines["pe"].set_data(ticks, self._pe)
        ax = self._axes[2]
        ax.relim()
        ax.autoscale_view()

        # 生死动态
        self._lines["births"].set_data(ticks, self._births)
        self._lines["deaths"].set_data(ticks, self._deaths)
        ax = self._axes[3]
        ax.relim()
        ax.autoscale_view()

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def finalize(self) -> None:
        """仿真结束后调用 — 保持窗口显示直到用户关闭"""
        if self._initialized and self._fig is not None:
            plt.ioff()
            print("  [实时监控] 仿真结束，关闭窗口以退出")
            plt.show()
