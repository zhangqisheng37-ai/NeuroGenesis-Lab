"""单元测试 — observability 模块"""
import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import pytest
from neurogenesis_lab.config import SCHEMA_VERSION
from neurogenesis_lab.encoding import Encoder, Event
from neurogenesis_lab.network import Network
from neurogenesis_lab.neuron import LIFNeuron, IzhikevichNeuron
from neurogenesis_lab.observability import (
    JsonlWriter,
    NetworkObserver,
    ObservabilityLogger,
    create_experiment_logger,
)


class TestJsonlWriter:
    def test_write_and_read(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        writer = JsonlWriter(path)
        writer.write([{"a": 1}, {"b": 2}])
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}

    def test_append(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        writer = JsonlWriter(path)
        writer.append({"x": 1})
        writer.append({"y": 2})
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        writer = JsonlWriter(path)
        writer.append({"test": True})
        assert path.exists()


class TestNetworkObserver:
    def _setup(self) -> tuple[Network, list[LIFNeuron], NetworkObserver]:
        neurons = [LIFNeuron() for _ in range(3)]
        net = Network(neurons={n.id: n for n in neurons})
        observer = NetworkObserver(net, debug_neuron_ids=[neurons[0].id])
        return net, neurons, observer

    def test_get_network_metrics(self) -> None:
        net, neurons, observer = self._setup()
        metrics = observer.get_network_metrics(1)
        assert metrics["tick"] == 1
        assert metrics["total_neurons"] == 3
        assert metrics["alive_neurons"] == 3
        assert "avg_energy" in metrics
        assert "avg_entropy" in metrics
        assert "active_ratio" in metrics

    def test_get_neuron_metrics(self) -> None:
        net, neurons, observer = self._setup()
        detail = observer.get_neuron_metrics(neurons[0].id)
        assert detail is not None
        assert detail["neuron_id"] == neurons[0].id
        assert "type" in detail
        assert "spike_count" in detail
        assert "v" in detail

    def test_get_neuron_metrics_unknown(self) -> None:
        _, _, observer = self._setup()
        assert observer.get_neuron_metrics("nonexistent") is None

    def test_get_learning_metrics(self) -> None:
        _, _, observer = self._setup()
        metrics = observer.get_learning_metrics()
        assert "total_spikes" in metrics
        assert "pe_moving_avg" in metrics

    def test_get_trend_metrics(self) -> None:
        _, _, observer = self._setup()
        snapshot = {"prediction_error": 0.5, "births": 1, "deaths": 0}
        trends = observer.get_trend_metrics(1, snapshot)
        assert "pe_moving_avg" in trends
        assert "birth_rate" in trends
        assert "death_rate" in trends

    def test_generate_report_has_schema_version(self) -> None:
        _, _, observer = self._setup()
        report = observer.generate_report(1)
        assert report["schema_version"] == SCHEMA_VERSION

    def test_debug_with_filter(self) -> None:
        neurons = [LIFNeuron(), IzhikevichNeuron()]
        net = Network(neurons={n.id: n for n in neurons})
        # 过滤追踪 IzhikevichNeuron
        observer = NetworkObserver(
            net,
            neuron_filter=lambda n: type(n).__name__ == "IzhikevichNeuron",
        )
        debug = observer.get_debug_info(1)
        tracked_types = [n["type"] for n in debug["tracked_neurons"]]
        assert "IzhikevichNeuron" in tracked_types

    def test_backward_compat_get_tick_metrics(self) -> None:
        _, _, observer = self._setup()
        metrics = observer.get_tick_metrics(1)
        assert "tick" in metrics


class TestObservabilityLogger:
    def test_log_tick_and_summary(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        neurons = [LIFNeuron()]
        net = Network(neurons={n.id: n for n in neurons})
        observer = NetworkObserver(net)
        logger = ObservabilityLogger(path, debug_mode=False)

        logger.log_tick(observer, 1)
        logger.log_summary(1, observer.get_network_metrics(1))

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "metrics" in record
        summary = json.loads(lines[1])
        assert summary["type"] == "summary"
        assert summary["schema_version"] == SCHEMA_VERSION

    def test_log_config(self, tmp_path: Path) -> None:
        path = tmp_path / "run.jsonl"
        logger = ObservabilityLogger(path)
        logger.log_config({"seed": 42, "ticks": 100})
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "config"
        assert record["schema_version"] == SCHEMA_VERSION

    def test_experiment_id_creates_subdir(self, tmp_path: Path) -> None:
        logger = ObservabilityLogger(
            tmp_path / "run.jsonl",
            experiment_id="exp_001",
        )
        assert "exp_001" in str(logger.get_output_path())


class TestCreateExperimentLogger:
    def test_creates_timestamped_dir(self, tmp_path: Path) -> None:
        logger = create_experiment_logger(tmp_path, "test_exp")
        path = logger.get_output_path()
        assert "test_exp" in str(path)
        assert path.name == "run.jsonl"


class TestRegressionRun:
    """回归测试 — 固定输入校验 jsonl 字段"""

    def test_standard_run_output(self, tmp_path: Path) -> None:
        random.seed(42)
        encoder = Encoder()
        a = LIFNeuron(tau_m=1.0, r_m=30.0, v_th=-64.0)
        b = LIFNeuron()
        net = Network(
            neurons={a.id: a, b.id: b},
            input_current_scale=5.0,
        )
        net.connect(a.id, b.id, weight=1.0, delay=1)

        tokens = encoder.encode_expr("1+1=2")
        events = encoder.to_events(tokens, src_id=a.id)
        net.enqueue(events)

        output_path = tmp_path / "regression.jsonl"
        observer = NetworkObserver(net)
        logger = ObservabilityLogger(output_path)

        for t in range(1, 11):
            snapshot = net.tick(t)
            logger.log_tick(observer, t, snapshot)

        logger.log_summary(10, observer.get_network_metrics(10))

        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 11, f"应有 10 tick + 1 summary，实际 {len(lines)} 行"

        # 校验每条记录的关键字段
        for line in lines[:-1]:
            record = json.loads(line)
            assert "schema_version" in record, "每条记录应含 schema_version"
            assert "metrics" in record, "每条记录应含 metrics"
            assert "learning" in record, "每条记录应含 learning"
            assert "trends" in record, "每条记录应含 trends"

        summary = json.loads(lines[-1])
        assert summary["type"] == "summary"
        assert summary["schema_version"] == SCHEMA_VERSION
