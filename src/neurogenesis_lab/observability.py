import json
from pathlib import Path
from typing import Iterable

from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import Neuron


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, records: Iterable[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, record: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class NetworkObserver:
    """网络级别的可观察性"""

    def __init__(self, network: Network, debug_neuron_ids: list[str] | None = None) -> None:
        self._network = network
        self._debug_neuron_ids = debug_neuron_ids or []

    def get_tick_metrics(self, tick: int) -> dict:
        """获取当前tick的网络指标"""
        neurons = list(self._network.neurons.values())
        alive_neurons = [n for n in neurons if n.alive]
        dead_neurons = [n for n in neurons if not n.alive]

        total_connections = sum(len(n.links) for n in neurons)
        avg_energy = sum(n.energy for n in alive_neurons) / len(alive_neurons) if alive_neurons else 0.0
        avg_novelty = (
            sum(n.states["novelty"] for n in alive_neurons) / len(alive_neurons)
            if alive_neurons
            else 0.0
        )
        avg_emotion = (
            sum(n.states["emotion"] for n in alive_neurons) / len(alive_neurons)
            if alive_neurons
            else 0.0
        )

        metrics = {
            "tick": tick,
            "total_neurons": len(neurons),
            "alive_neurons": len(alive_neurons),
            "dead_neurons": len(dead_neurons),
            "total_connections": total_connections,
            "avg_connections": total_connections / len(neurons) if neurons else 0.0,
            "avg_energy": avg_energy,
            "queue_size": len(self._network.queue),
            "avg_novelty": avg_novelty,
            "avg_emotion": avg_emotion,
        }

        return metrics

    def get_neuron_detail(self, neuron_id: str) -> dict | None:
        """获取单个神经元的详细信息"""
        if neuron_id not in self._network.neurons:
            return None

        neuron = self._network.neurons[neuron_id]
        recent_tokens = [event.token_id for event in list(neuron.mem)[-10:]]
        observations = neuron.observe()

        return {
            "neuron_id": neuron_id,
            "pos": neuron.pos,
            "energy": neuron.energy,
            "alive": neuron.alive,
            "connections": len(neuron.links),
            "memory_size": len(neuron.mem),
            "recent_tokens": recent_tokens,
            "top_transitions": observations["transition_counts"][:5],
            "states": observations["states"],
        }

    def get_debug_info(self, tick: int) -> dict:
        """获取调试信息（追踪指定神经元）"""
        debug_info = {"tick": tick, "tracked_neurons": []}

        for neuron_id in self._debug_neuron_ids:
            detail = self.get_neuron_detail(neuron_id)
            if detail:
                debug_info["tracked_neurons"].append(detail)

        return debug_info

    def generate_report(self, tick: int, include_debug: bool = False) -> dict:
        """生成完整报告"""
        report = {
            "metrics": self.get_tick_metrics(tick),
        }

        if include_debug and self._debug_neuron_ids:
            report["debug"] = self.get_debug_info(tick)

        return report


class ObservabilityLogger:
    """可观察性日志记录器"""

    def __init__(self, output_path: Path, debug_mode: bool = False) -> None:
        self._writer = JsonlWriter(output_path)
        self._debug_mode = debug_mode
        self._output_path = output_path

    def log_tick(self, observer: NetworkObserver, tick: int) -> None:
        """记录单个tick的信息"""
        report = observer.generate_report(tick, include_debug=self._debug_mode)
        self._writer.append(report)

    def log_summary(self, total_ticks: int, final_metrics: dict) -> None:
        """记录总结信息"""
        summary = {
            "type": "summary",
            "total_ticks": total_ticks,
            "final_metrics": final_metrics,
        }
        self._writer.append(summary)

    def get_output_path(self) -> Path:
        return self._output_path
