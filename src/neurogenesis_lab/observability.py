import json
import gzip
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import Neuron


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, records: Iterable[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _open(self, mode: str):
        if self._path.suffix == ".gz":
            return gzip.open(self._path, mode + "t", encoding="utf-8")
        return self._path.open(mode, encoding="utf-8")


class NetworkObserver:
    """网络级别的可观察性 2.0"""

    def __init__(
        self,
        network: Network,
        debug_neuron_ids: list[str] | None = None,
        neuron_filter: Callable[[Neuron], bool] | None = None,
    ) -> None:
        self._network = network
        self._debug_neuron_ids = debug_neuron_ids or []
        self._neuron_filter = neuron_filter
        # 趋势跟踪
        self._pe_history: list[float] = []
        self._alive_history: list[int] = []
        self._birth_history: list[int] = []
        self._death_history: list[int] = []
        self._energy_history: list[float] = []

    def get_network_metrics(self, tick: int) -> dict:
        """获取 network_metrics"""
        neurons = list(self._network.neurons.values())
        alive_neurons = [n for n in neurons if n.alive]

        total_connections = sum(len(n.links) for n in neurons)
        avg_energy = sum(n.energy for n in alive_neurons) / len(alive_neurons) if alive_neurons else 0.0
        avg_novelty = sum(n.states["novelty"] for n in alive_neurons) / len(alive_neurons) if alive_neurons else 0.0
        avg_emotion = sum(n.states["emotion"] for n in alive_neurons) / len(alive_neurons) if alive_neurons else 0.0
        avg_entropy = sum(n.states["entropy"] for n in alive_neurons) / len(alive_neurons) if alive_neurons else 0.0

        self._alive_history.append(len(alive_neurons))
        self._energy_history.append(avg_energy)

        return {
            "tick": tick,
            "total_neurons": len(neurons),
            "alive_neurons": len(alive_neurons),
            "dead_neurons": len(neurons) - len(alive_neurons),
            "total_connections": total_connections,
            "avg_connections": total_connections / len(neurons) if neurons else 0.0,
            "mean_energy": avg_energy,
            "queue_size": len(self._network.queue),
            "avg_novelty": avg_novelty,
            "avg_emotion": avg_emotion,
            "avg_entropy": avg_entropy,
            "active_ratio": self._network.active_neuron_ratio(),
            "corpse_count": len(self._network.corpses),
        }

    def get_neuron_metrics(self, neuron_id: str) -> dict | None:
        """获取 neuron_metrics"""
        if neuron_id not in self._network.neurons:
            return None
        neuron = self._network.neurons[neuron_id]
        recent_tokens = [event.token_id for event in list(neuron.mem)[-10:]]
        observations = neuron.observe()
        return {
            "neuron_id": neuron_id,
            "type": type(neuron).__name__,
            "pos": neuron.pos,
            "energy": neuron.energy,
            "alive": neuron.alive,
            "connections": len(neuron.links),
            "memory_size": len(neuron.mem),
            "recent_tokens": recent_tokens,
            "top_transitions": observations["transition_counts"][:5],
            "states": observations["states"],
            "spike_count": neuron.spike_count,
            "propagation_count": neuron.propagation_count,
            "v": neuron.v,
        }

    def get_learning_metrics(self) -> dict:
        """获取 learning_metrics"""
        return {
            "total_spikes": self._network.total_spikes,
            "total_propagations": self._network.total_propagations,
            "pe_moving_avg": self._network.pe_moving_average(),
        }

    def get_trend_metrics(self, tick: int, snapshot: dict | None = None) -> dict:
        """获取趋势指标"""
        if snapshot:
            self._pe_history.append(snapshot.get("prediction_error", 0.0))
            self._birth_history.append(snapshot.get("births", 0))
            self._death_history.append(snapshot.get("deaths", 0))

        window = 10
        pe_window = self._pe_history[-window:] if self._pe_history else [0.0]
        birth_rate = sum(self._birth_history[-window:]) / min(window, len(self._birth_history)) if self._birth_history else 0.0
        death_rate = sum(self._death_history[-window:]) / min(window, len(self._death_history)) if self._death_history else 0.0
        alive_window = self._alive_history[-window:] if self._alive_history else [0]

        return {
            "tick": tick,
            "pe_moving_avg": sum(pe_window) / len(pe_window),
            "birth_rate": birth_rate,
            "death_rate": death_rate,
            "alive_trend": alive_window,
            "effective_propagation_rate": (
                self._network.total_propagations / max(1, self._network.total_spikes)
            ),
        }

    # 向后兼容
    def get_tick_metrics(self, tick: int) -> dict:
        return self.get_network_metrics(tick)

    def get_neuron_detail(self, neuron_id: str) -> dict | None:
        return self.get_neuron_metrics(neuron_id)

    def get_debug_info(self, tick: int) -> dict:
        """获取调试信息（追踪指定神经元，支持条件过滤）"""
        debug_info: dict[str, Any] = {"tick": tick, "tracked_neurons": []}

        neuron_ids = list(self._debug_neuron_ids)
        # 如果有过滤器，追踪符合条件的神经元
        if self._neuron_filter:
            for nid, neuron in self._network.neurons.items():
                if self._neuron_filter(neuron) and nid not in neuron_ids:
                    neuron_ids.append(nid)

        for neuron_id in neuron_ids:
            detail = self.get_neuron_metrics(neuron_id)
            if detail:
                debug_info["tracked_neurons"].append(detail)

        return debug_info

    def generate_report(self, tick: int, include_debug: bool = False, snapshot: dict | None = None) -> dict:
        """生成完整报告"""
        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "metrics": self.get_network_metrics(tick),
            "learning": self.get_learning_metrics(),
            "trends": self.get_trend_metrics(tick, snapshot),
        }

        if include_debug and (self._debug_neuron_ids or self._neuron_filter):
            report["debug"] = self.get_debug_info(tick)

        return report


class ObservabilityLogger:
    """可观察性日志记录器 2.0 — 支持实验分目录"""

    def __init__(
        self,
        output_path: Path,
        debug_mode: bool = False,
        experiment_id: str | None = None,
        log_interval: int = 1,
        max_file_size: int = 0,
    ) -> None:
        if experiment_id:
            # 按实验编号自动分目录
            output_path = output_path.parent / experiment_id / output_path.name
        self._base_output_path = output_path
        self._output_path = output_path
        self._segment_index = 0
        self._writer = JsonlWriter(self._output_path)
        self._debug_mode = debug_mode
        self._log_interval = max(1, int(log_interval))
        self._max_file_size = max(0, int(max_file_size))

    def _segment_path(self, segment_index: int) -> Path:
        if segment_index <= 0:
            return self._base_output_path
        name = self._base_output_path.name
        if name.endswith(".jsonl.gz"):
            stem = name[:-9]
            return self._base_output_path.with_name(f"{stem}_{segment_index:04d}.jsonl.gz")
        if name.endswith(".jsonl"):
            stem = name[:-6]
            return self._base_output_path.with_name(f"{stem}_{segment_index:04d}.jsonl")
        return self._base_output_path.with_name(f"{name}_{segment_index:04d}")

    def _rotate_if_needed(self) -> None:
        if self._max_file_size <= 0:
            return
        if not self._output_path.exists():
            return
        if self._output_path.stat().st_size < self._max_file_size:
            return
        self._segment_index += 1
        self._output_path = self._segment_path(self._segment_index)
        self._writer = JsonlWriter(self._output_path)

    def log_config(self, config_dict: dict) -> None:
        """记录实验配置"""
        self._rotate_if_needed()
        record = {"type": "config", "schema_version": SCHEMA_VERSION, **config_dict}
        self._writer.append(record)

    def log_tick(self, observer: NetworkObserver, tick: int, snapshot: dict | None = None) -> None:
        """记录单个tick的信息"""
        if tick % self._log_interval != 0:
            return
        self._rotate_if_needed()
        report = observer.generate_report(tick, include_debug=self._debug_mode, snapshot=snapshot)
        self._writer.append(report)

    def log_summary(self, total_ticks: int, final_metrics: dict) -> None:
        """记录总结信息"""
        self._rotate_if_needed()
        summary = {
            "type": "summary",
            "schema_version": SCHEMA_VERSION,
            "total_ticks": total_ticks,
            "final_metrics": final_metrics,
        }
        self._writer.append(summary)

    def get_output_path(self) -> Path:
        return self._output_path


def create_experiment_logger(
    base_dir: Path,
    experiment_name: str,
    debug_mode: bool = False,
    log_interval: int = 1,
    max_file_size: int = 0,
    gzip_output: bool = False,
) -> ObservabilityLogger:
    """创建实验日志记录器 — 自动分目录防覆盖"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_id = f"{experiment_name}_{timestamp}"
    filename = "run.jsonl.gz" if gzip_output else "run.jsonl"
    output_path = base_dir / exp_id / filename
    return ObservabilityLogger(
        output_path,
        debug_mode=debug_mode,
        log_interval=log_interval,
        max_file_size=max_file_size,
    )
