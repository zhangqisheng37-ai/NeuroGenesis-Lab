"""单元测试 — CLI 入口分发与信息输出"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import pytest

import neurogenesis_lab.__main__ as cli
from neurogenesis_lab.config import SimConfig


def test_cmd_info_outputs_version_and_schema(capsys) -> None:
    cli.cmd_info(cli.argparse.Namespace())
    out = capsys.readouterr().out
    assert "NeuroGenesis Lab v" in out
    assert "Schema Version" in out
    assert "Rust Core" in out
    assert "默认配置" in out


def test_main_dispatches_info(monkeypatch) -> None:
    called = {"info": False}

    def fake_info(args) -> None:
        called["info"] = True

    monkeypatch.setattr(cli, "cmd_info", fake_info)
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "info"])

    cli.main()

    assert called["info"] is True


def test_main_dispatches_run(monkeypatch) -> None:
    called = {"run": False}

    def fake_run(args) -> None:
        called["run"] = True

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "run"])

    cli.main()

    assert called["run"] is True


def test_main_run_accepts_checkpoint_args(monkeypatch) -> None:
    captured = {"args": None}

    def fake_run(args) -> None:
        captured["args"] = args

    monkeypatch.setattr(cli, "cmd_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neurogenesis-lab",
            "run",
            "--ticks",
            "20",
            "--checkpoint-interval",
            "5",
            "--checkpoint-dir",
            "outputs/checkpoints",
            "--resume-from",
            "outputs/checkpoints/checkpoint_tick_100.pkl",
            "--background-current",
            "0.5",
            "--spontaneous-spike-rate",
            "0.15",
            "--low-activation-threshold",
            "4",
        ],
    )

    cli.main()

    assert captured["args"] is not None
    assert captured["args"].checkpoint_interval == 5
    assert captured["args"].checkpoint_dir == "outputs/checkpoints"
    assert captured["args"].resume_from == "outputs/checkpoints/checkpoint_tick_100.pkl"
    assert captured["args"].background_current == 0.5
    assert captured["args"].spontaneous_spike_rate == 0.15
    assert captured["args"].low_activation_threshold == 4


def test_main_dispatches_benchmark(monkeypatch) -> None:
    called = {"benchmark": False}

    def fake_benchmark(args) -> None:
        called["benchmark"] = True

    monkeypatch.setattr(cli, "cmd_benchmark", fake_benchmark)
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "benchmark"])

    cli.main()

    assert called["benchmark"] is True


def test_main_without_command_prints_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab"])

    cli.main()

    out = capsys.readouterr().out
    assert "可用命令" in out
    assert "run" in out
    assert "benchmark" in out
    assert "info" in out


def test_cmd_run_rejects_invalid_config(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_build_sim_config", lambda args: SimConfig(ticks=0))

    with pytest.raises(SystemExit) as exc:
        cli.cmd_run(cli.argparse.Namespace())

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "配置验证失败" in out
    assert "ticks" in out


def test_cmd_run_smoke_creates_output(tmp_path: Path, capsys) -> None:
    args = cli.argparse.Namespace(
        ticks=3,
        seed=42,
        dt=1.0,
        neurons=4,
        expr="1+1=2",
        input_weight=60.0,
        stim_interval=1,
        output=str(tmp_path),
        config=None,
        debug=False,
        log_interval=1,
        gzip_output=False,
        max_file_size=0,
    )

    cli.cmd_run(args)

    out = capsys.readouterr().out
    assert "完成!" in out
    assert "总放电" in out
    assert list(tmp_path.glob("run_*/run.jsonl"))


def test_cmd_benchmark_smoke(capsys) -> None:
    args = cli.argparse.Namespace(neurons=8, ticks=3, seed=7, rust_size=64)
    cli.cmd_benchmark(args)

    out = capsys.readouterr().out
    assert "性能基准测试" in out
    assert "结果" in out
    assert "对照指标表" in out
    assert "Rust并行LIF" in out
    assert "alive_neurons" in out
    assert "event_routing_trace" in out


def test_cmd_benchmark_comparability_labels_are_stable(capsys) -> None:
    args = cli.argparse.Namespace(neurons=8, ticks=3, seed=7, rust_size=64)
    cli.cmd_benchmark(args)

    out = capsys.readouterr().out
    assert "可比" in out
    assert "不可比" in out
    assert "部分可比" in out


def test_main_run_smoke_with_checkpoint(tmp_path: Path, monkeypatch, capsys) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neurogenesis-lab",
            "run",
            "--ticks",
            "3",
            "--seed",
            "42",
            "--neurons",
            "4",
            "--output",
            str(tmp_path),
            "--checkpoint-interval",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )

    cli.main()

    out = capsys.readouterr().out
    assert "完成!" in out
    assert list(tmp_path.glob("run_*/run.jsonl"))
    assert list(checkpoint_dir.glob("checkpoint_tick_*.pkl"))


def test_main_benchmark_smoke_with_argv(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "neurogenesis-lab",
            "benchmark",
            "--neurons",
            "8",
            "--ticks",
            "3",
            "--seed",
            "7",
            "--rust-size",
            "64",
        ],
    )

    cli.main()

    out = capsys.readouterr().out
    assert "性能基准测试" in out
    assert "结果" in out
