import json
import gzip
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import Neuron

_logger = logging.getLogger(__name__)


class JsonlWriter:
    """JSONL 文件写入器 — 支持普通文本和 gzip 压缩格式。

    根据文件后缀自动选择写入模式：``.gz`` 使用 gzip 压缩，否则纯文本。

    Args:
        path: 输出文件路径。
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, records: Iterable[dict]) -> None:
        """覆盖写入多条记录（清空原有内容）。

        Args:
            records: 待写入的字典记录可迭代对象。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append(self, record: dict) -> None:
        """追加写入单条记录。

        Args:
            record: 待写入的字典记录。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def append_many(self, records: Iterable[dict]) -> None:
        """追加写入多条记录。

        Args:
            records: 待写入的字典记录可迭代对象。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._open("a") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _open(self, mode: str):
        if self._path.suffix == ".gz":
            return gzip.open(self._path, mode + "t", encoding="utf-8")
        return self._path.open(mode, encoding="utf-8")


class AsyncJsonlWriter:
    """异步双缓冲写盘器：主线程写入 Buffer A，后台线程落盘 Buffer B

    Args:
        path: 输出文件路径。
        buffer_size: 缓冲区满阈值（记录条数），达到后自动切换缓冲区。
        on_error: 写入失败时的回调函数，签名 ``(exception, lost_records)``。
            若未提供，则仅记录警告日志。
    """

    _WRITER_POLL_INTERVAL = 0.2

    def __init__(
        self,
        path: Path,
        buffer_size: int = 256,
        on_error: Callable[[Exception, list[dict]], None] | None = None,
    ) -> None:
        self._writer = JsonlWriter(path)
        self._buffer_size = max(1, int(buffer_size))
        self._on_error = on_error
        self._lost_records: int = 0
        self._buffers: list[list[dict]] = [[], []]
        self._active_idx = 0
        self._pending_idx: int | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._drain_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="observability-async-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def lost_records(self) -> int:
        """写入失败导致丢失的记录总数。"""
        return self._lost_records

    def append(self, record: dict) -> None:
        """将单条记录追加到活跃缓冲区。

        缓冲区满时自动切换并触发后台落盘。

        Args:
            record: 待写入的字典记录。

        Raises:
            RuntimeError: writer 已关闭时抛出。
        """
        with self._cond:
            if self._closed:
                raise RuntimeError("writer already closed")
            self._buffers[self._active_idx].append(record)
            self._promote_if_needed_locked(force=False)

    def write(self, records: Iterable[dict]) -> None:
        """逐条追加写入多条记录。

        Args:
            records: 待写入的字典记录可迭代对象。
        """
        for record in records:
            self.append(record)

    def close(self) -> None:
        """关闭写入器 — 刷新剩余缓冲区并等待后台线程退出。"""
        with self._cond:
            if self._closed:
                return
            # 将剩余数据提升为待写入
            self._promote_if_needed_locked(force=True)
            self._closed = True
            self._cond.notify_all()
        # 等待后台线程排空并退出，使用 Event 替代轮询
        self._drain_event.wait(timeout=5.0)
        self._thread.join(timeout=2.0)

    def _promote_if_needed_locked(self, force: bool) -> None:
        if self._pending_idx is not None:
            return
        active = self._buffers[self._active_idx]
        if not active:
            return
        if (not force) and len(active) < self._buffer_size:
            return
        self._pending_idx = self._active_idx
        self._active_idx = 1 - self._active_idx
        self._buffers[self._active_idx] = []
        self._cond.notify_all()

    def _run(self) -> None:
        try:
            while True:
                with self._cond:
                    while self._pending_idx is None and not self._closed:
                        self._cond.wait(timeout=AsyncJsonlWriter._WRITER_POLL_INTERVAL)
                        self._promote_if_needed_locked(force=True)

                    if self._pending_idx is None and self._closed:
                        # 最后一次尝试提升残留数据
                        self._promote_if_needed_locked(force=True)
                        if self._pending_idx is None:
                            return

                    pending_idx = self._pending_idx
                    self._pending_idx = None
                    batch = self._buffers[pending_idx] if pending_idx is not None else []
                    if pending_idx is not None:
                        self._buffers[pending_idx] = []
                    self._cond.notify_all()

                if batch:
                    self._flush_batch(batch)
        finally:
            self._drain_event.set()

    def _flush_batch(self, batch: list[dict]) -> None:
        """落盘一批记录，失败时触发错误回调。"""
        try:
            self._writer.append_many(batch)
        except Exception as exc:
            self._lost_records += len(batch)
            if self._on_error is not None:
                try:
                    self._on_error(exc, batch)
                except Exception:
                    pass
            else:
                _logger.warning(
                    "AsyncJsonlWriter 写入失败，丢失 %d 条记录: %s",
                    len(batch),
                    exc,
                )


class NetworkObserver:
    """网络级别的可观察性 — 采集指标、趋势和调试信息。

    Args:
        network: 被观察的 Network 实例。
        debug_neuron_ids: 需要追踪的神经元 ID 列表。
        neuron_filter: 可选的神经元过滤函数，满足条件的神经元也会被追踪。
    """

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
        """获取网络级聚合指标。

        Args:
            tick: 当前 tick 编号。

        Returns:
            包含存活数、连接数、平均能量、平均新颖度等指标的字典。
        """
        neurons = list(self._network.neurons.values())
        alive_neurons = [n for n in neurons if n.alive]

        total_connections = sum(len(n.links) for n in neurons)
        avg_energy = (
            sum(n.energy for n in alive_neurons) / len(alive_neurons)
            if alive_neurons
            else 0.0
        )
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
        avg_entropy = (
            sum(n.states["entropy"] for n in alive_neurons) / len(alive_neurons)
            if alive_neurons
            else 0.0
        )

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
        """获取单个神经元的详细指标。

        Args:
            neuron_id: 目标神经元 ID。

        Returns:
            神经元指标字典，神经元不存在时返回 None。
        """
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
        """获取学习相关指标（放电数、传播数、PE 均值）。

        Returns:
            包含 total_spikes、total_propagations、pe_moving_avg 的字典。
        """
        return {
            "total_spikes": self._network.total_spikes,
            "total_propagations": self._network.total_propagations,
            "pe_moving_avg": self._network.pe_moving_average(),
        }

    def get_trend_metrics(self, tick: int, snapshot: dict | None = None) -> dict:
        """获取趋势指标（滑动窗口统计）。

        Args:
            tick: 当前 tick 编号。
            snapshot: 可选的 TickSnapshot，用于更新历史记录。

        Returns:
            包含 PE 均值、出生率、死亡率、存活趋势等的字典。
        """
        if snapshot:
            self._pe_history.append(snapshot.get("prediction_error", 0.0))
            self._birth_history.append(snapshot.get("births", 0))
            self._death_history.append(snapshot.get("deaths", 0))

        window = 10
        pe_window = self._pe_history[-window:] if self._pe_history else [0.0]
        birth_rate = (
            sum(self._birth_history[-window:]) / min(window, len(self._birth_history))
            if self._birth_history
            else 0.0
        )
        death_rate = (
            sum(self._death_history[-window:]) / min(window, len(self._death_history))
            if self._death_history
            else 0.0
        )
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
        """获取调试信息 — 追踪指定神经元，支持条件过滤。

        Args:
            tick: 当前 tick 编号。

        Returns:
            包含 tick 和 tracked_neurons 列表的字典。
        """
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

    def generate_report(
        self, tick: int, include_debug: bool = False, snapshot: dict | None = None
    ) -> dict:
        """生成完整的可观察性报告。

        Args:
            tick: 当前 tick 编号。
            include_debug: 是否包含调试信息，默认 False。
            snapshot: 可选的 TickSnapshot，传递给趋势指标。

        Returns:
            包含 metrics、learning、trends（可选 debug）的完整报告字典。
        """
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
    """可观察性日志记录器 — 支持采样、轮转和异步写入。

    Args:
        output_path: 输出文件路径。
        debug_mode: 是否记录调试信息，默认 False。
        experiment_id: 实验编号，非空时自动创建子目录。
        log_interval: 日志采样间隔（每 N tick 记录一次），默认 1。
        max_file_size: 单文件最大字节数，超出后自动轮转，0 表示不限，默认 0。
        async_mode: 是否启用异步双缓冲写入，默认 False。
        log_buffer_size: 异步模式缓冲区大小（记录条数），默认 256。
    """

    def __init__(
        self,
        output_path: Path,
        debug_mode: bool = False,
        experiment_id: str | None = None,
        log_interval: int = 1,
        max_file_size: int = 0,
        async_mode: bool = False,
        log_buffer_size: int = 256,
    ) -> None:
        if experiment_id:
            # 按实验编号自动分目录
            output_path = output_path.parent / experiment_id / output_path.name
        self._base_output_path = output_path
        self._output_path = output_path
        self._segment_index = 0
        self._async_mode = bool(async_mode)
        self._log_buffer_size = max(1, int(log_buffer_size))
        self._writer: JsonlWriter | AsyncJsonlWriter = self._create_writer()
        self._debug_mode = debug_mode
        self._log_interval = max(1, int(log_interval))
        self._max_file_size = max(0, int(max_file_size))

    def _create_writer(self) -> JsonlWriter | AsyncJsonlWriter:
        if self._async_mode:
            return AsyncJsonlWriter(
                self._output_path, buffer_size=self._log_buffer_size
            )
        return JsonlWriter(self._output_path)

    def _close_writer(self) -> None:
        close_fn = getattr(self._writer, "close", None)
        if callable(close_fn):
            close_fn()

    def _segment_path(self, segment_index: int) -> Path:
        if segment_index <= 0:
            return self._base_output_path
        name = self._base_output_path.name
        if name.endswith(".jsonl.gz"):
            stem = name[:-9]
            return self._base_output_path.with_name(
                f"{stem}_{segment_index:04d}.jsonl.gz"
            )
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
        self._close_writer()
        self._output_path = self._segment_path(self._segment_index)
        self._writer = self._create_writer()

    def log_config(self, config_dict: dict) -> None:
        """记录实验配置"""
        self._rotate_if_needed()
        record = {"type": "config", "schema_version": SCHEMA_VERSION, **config_dict}
        self._writer.append(record)

    def log_tick(
        self, observer: NetworkObserver, tick: int, snapshot: dict | None = None
    ) -> None:
        """记录单个 tick 的可观察性报告（受 log_interval 采样控制）。

        Args:
            observer: NetworkObserver 实例。
            tick: 当前 tick 编号。
            snapshot: 可选的 TickSnapshot，传递给报告生成。
        """
        if tick % self._log_interval != 0:
            return
        self._rotate_if_needed()
        report = observer.generate_report(
            tick, include_debug=self._debug_mode, snapshot=snapshot
        )
        self._writer.append(report)

    def log_summary(self, total_ticks: int, final_metrics: dict) -> None:
        """记录仿真结束时的总结信息。

        Args:
            total_ticks: 总运行 tick 数。
            final_metrics: 最终指标字典。
        """
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

    def close(self) -> None:
        self._close_writer()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def create_experiment_logger(
    base_dir: Path,
    experiment_name: str,
    debug_mode: bool = False,
    log_interval: int = 1,
    max_file_size: int = 0,
    gzip_output: bool = False,
    async_mode: bool = False,
    log_buffer_size: int = 256,
) -> ObservabilityLogger:
    """创建实验日志记录器 — 自动按时间戳分目录防覆盖。

    Args:
        base_dir: 输出根目录。
        experiment_name: 实验名称前缀。
        debug_mode: 是否记录调试信息，默认 False。
        log_interval: 日志采样间隔，默认 1。
        max_file_size: 单文件最大字节数，0 表示不限，默认 0。
        gzip_output: 是否使用 gzip 压缩输出，默认 False。
        async_mode: 是否启用异步写入，默认 False。
        log_buffer_size: 异步缓冲区大小，默认 256。

    Returns:
        配置好的 ObservabilityLogger 实例。
    """
    # 使用毫秒精度时间戳，避免快速连续创建时冲突
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    milliseconds = int((time.time() % 1) * 1000)
    exp_id = f"{experiment_name}_{timestamp}_{milliseconds:03d}"
    filename = "run.jsonl.gz" if gzip_output else "run.jsonl"
    output_path = base_dir / exp_id / filename
    return ObservabilityLogger(
        output_path,
        debug_mode=debug_mode,
        log_interval=log_interval,
        max_file_size=max_file_size,
        async_mode=async_mode,
        log_buffer_size=log_buffer_size,
    )
