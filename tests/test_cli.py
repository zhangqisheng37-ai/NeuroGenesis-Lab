"""单元测试 — CLI 入口分发、配置构建与实验管理子命令（原三个 CLI 测试文件合并）"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import neurogenesis_lab.cli.benchmark as cli_bench
import neurogenesis_lab.cli.compare as cli_compare
import neurogenesis_lab.cli.delete as cli_delete
import neurogenesis_lab.cli.info as cli_info
import neurogenesis_lab.cli.list as cli_list
import neurogenesis_lab.cli.parser as cli_parser
import neurogenesis_lab.cli.run as cli_run
from neurogenesis_lab.cli.config_builder import build_sim_config
from neurogenesis_lab.config import SimConfig
from neurogenesis_lab.experiments import ExperimentRecord, ExperimentStore


def test_cmd_info_outputs_version_and_schema(capsys) -> None:
    cli_info.cmd_info(argparse.Namespace())
    out = capsys.readouterr().out
    assert "NeuroGenesis Lab v" in out
    assert "Schema Version" in out
    assert "Rust Core" in out
    assert "默认配置" in out


def test_main_dispatches_info(monkeypatch) -> None:
    called = {"info": False}

    def fake_info(args) -> None:
        called["info"] = True

    monkeypatch.setattr(cli_parser, "cmd_info", fake_info)
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "info"])

    cli_parser.main()

    assert called["info"] is True


def test_main_dispatches_run(monkeypatch) -> None:
    called = {"run": False}

    def fake_run(args) -> None:
        called["run"] = True

    monkeypatch.setattr(cli_parser, "cmd_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "run"])

    cli_parser.main()

    assert called["run"] is True


def test_main_run_accepts_checkpoint_args(monkeypatch) -> None:
    captured = {"args": None}

    def fake_run(args) -> None:
        captured["args"] = args

    monkeypatch.setattr(cli_parser, "cmd_run", fake_run)
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

    cli_parser.main()

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

    monkeypatch.setattr(cli_parser, "cmd_benchmark", fake_benchmark)
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "benchmark"])

    cli_parser.main()

    assert called["benchmark"] is True


def test_main_without_command_prints_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab"])

    cli_parser.main()

    out = capsys.readouterr().out
    assert "可用命令" in out
    assert "run" in out
    assert "benchmark" in out
    assert "info" in out


def test_cmd_run_rejects_invalid_config(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli_run, "build_sim_config", lambda args: SimConfig(ticks=0))

    with pytest.raises(SystemExit) as exc:
        cli_run.cmd_run(argparse.Namespace())

    out = capsys.readouterr().out
    assert exc.value.code == 1
    assert "配置验证失败" in out
    assert "ticks" in out


def test_cmd_run_smoke_creates_output(tmp_path: Path, capsys) -> None:
    args = argparse.Namespace(
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
        async_log=False,
        log_buffer_size=256,
        visualize=False,
        plot_interval=5,
        checkpoint_interval=0,
        checkpoint_dir=str(tmp_path),
        resume_from=None,
        background_current=None,
        spontaneous_spike_rate=None,
        low_activation_threshold=None,
    )

    cli_run.cmd_run(args)

    out = capsys.readouterr().out
    assert "完成!" in out
    assert "总放电" in out
    assert list(tmp_path.glob("run_*/run.jsonl"))


def test_cmd_benchmark_smoke(capsys) -> None:
    args = argparse.Namespace(neurons=8, ticks=3, seed=7, rust_size=64)
    cli_bench.cmd_benchmark(args)

    out = capsys.readouterr().out
    assert "性能基准测试" in out
    assert "结果" in out
    assert "对照指标表" in out
    assert "Rust并行LIF" in out
    assert "alive_neurons" in out
    assert "event_routing_trace" in out


def test_cmd_benchmark_comparability_labels_are_stable(capsys) -> None:
    args = argparse.Namespace(neurons=8, ticks=3, seed=7, rust_size=64)
    cli_bench.cmd_benchmark(args)

    out = capsys.readouterr().out
    assert "可比" in out
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

    cli_parser.main()

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

    cli_parser.main()

    out = capsys.readouterr().out
    assert "性能基准测试" in out
    assert "结果" in out


# ======================================================================
# 合并自 test_cli_config.py：单元测试 — CLI 配置映射
# ======================================================================


def test_build_sim_config_missing_sections_uses_defaults(tmp_path: Path) -> None:
    cfg_file = tmp_path / "minimal.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = 120
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=None,
        seed=None,
        dt=None,
        background_current=None,
        spontaneous_spike_rate=None,
        low_activation_threshold=None,
    )
    cfg = build_sim_config(args)

    assert cfg.ticks == 120
    # 缺失 section 应回退默认值
    assert cfg.neuron.v_rest == -65.0
    assert cfg.network.max_neurons == 50
    assert cfg.evolution.min_neurons == 2


def test_build_sim_config_nested_override_partial_fitness_weights(tmp_path: Path) -> None:
    cfg_file = tmp_path / "fitness.toml"
    cfg_file.write_text(
        """
[evolution.fitness_weights]
propagation = 1.8
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=None,
        seed=None,
        dt=None,
        background_current=None,
        spontaneous_spike_rate=None,
        low_activation_threshold=None,
    )
    cfg = build_sim_config(args)

    assert cfg.evolution.fitness_weights["propagation"] == 1.8
    # 其余键应保留默认值
    assert cfg.evolution.fitness_weights["energy_efficiency"] == 0.5
    assert cfg.evolution.fitness_weights["prediction_hit"] == 0.8


def test_build_sim_config_invalid_numeric_value_raises(tmp_path: Path) -> None:
    cfg_file = tmp_path / "invalid.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = "abc"
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=None,
        seed=None,
        dt=None,
        background_current=None,
        spontaneous_spike_rate=None,
        low_activation_threshold=None,
    )

    with pytest.raises((TypeError, ValueError)):
        build_sim_config(args)


def test_build_sim_config_maps_all_toml_sections(tmp_path: Path) -> None:
    cfg_file = tmp_path / "experiment.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = 222
seed = 99
dt = 0.5

[neuron]
v_rest = -66.0
v_reset = -64.0
v_th = -54.0
tau_m = 12.0
r_m = 1.2
spike_cost = 0.03
input_gain = 0.08
mem_size = 77

[network]
reproduction_threshold = 3.2
reproduction_share = 0.33
mutation_scale = 0.07
energy_decay = 0.004
max_neurons = 88
max_links = 333
corpse_ttl = 66
scavenger_radius = 3.3
input_current_scale = 2.2
background_current = 0.6
spontaneous_spike_rate = 0.12
low_activation_threshold = 3
move_cost = 0.02

[evolution]
mutation_amplitude = 0.09
energy_budget_per_tick = 1.5
congestion_penalty = 0.21
min_neurons = 0

[evolution.fitness_weights]
propagation = 1.1
energy_efficiency = 0.7
prediction_hit = 0.9
""",
        encoding="utf-8",
    )

    args = Namespace(config=str(cfg_file), ticks=None, seed=None, dt=None)
    cfg = build_sim_config(args)

    assert cfg.ticks == 222
    assert cfg.seed == 99
    assert cfg.dt == 0.5

    assert cfg.neuron.v_rest == -66.0
    assert cfg.neuron.v_reset == -64.0
    assert cfg.neuron.v_th == -54.0
    assert cfg.neuron.tau_m == 12.0
    assert cfg.neuron.r_m == 1.2
    assert cfg.neuron.spike_cost == 0.03
    assert cfg.neuron.input_gain == 0.08
    assert cfg.neuron.mem_size == 77

    assert cfg.network.reproduction_threshold == 3.2
    assert cfg.network.reproduction_share == 0.33
    assert cfg.network.mutation_scale == 0.07
    assert cfg.network.energy_decay == 0.004
    assert cfg.network.max_neurons == 88
    assert cfg.network.max_links == 333
    assert cfg.network.corpse_ttl == 66
    assert cfg.network.scavenger_radius == 3.3
    assert cfg.network.input_current_scale == 2.2
    assert cfg.network.background_current == 0.6
    assert cfg.network.spontaneous_spike_rate == 0.12
    assert cfg.network.low_activation_threshold == 3
    assert cfg.network.move_cost == 0.02

    assert cfg.evolution.mutation_amplitude == 0.09
    assert cfg.evolution.energy_budget_per_tick == 1.5
    assert cfg.evolution.congestion_penalty == 0.21
    assert cfg.evolution.min_neurons == 0
    assert cfg.evolution.fitness_weights["propagation"] == 1.1
    assert cfg.evolution.fitness_weights["energy_efficiency"] == 0.7
    assert cfg.evolution.fitness_weights["prediction_hit"] == 0.9


def test_build_sim_config_cli_overrides_toml(tmp_path: Path) -> None:
    cfg_file = tmp_path / "experiment.toml"
    cfg_file.write_text(
        """
[simulation]
ticks = 100
seed = 7
dt = 1.0
""",
        encoding="utf-8",
    )

    args = Namespace(
        config=str(cfg_file),
        ticks=300,
        seed=123,
        dt=0.25,
        background_current=0.4,
        spontaneous_spike_rate=0.2,
        low_activation_threshold=5,
    )
    cfg = build_sim_config(args)

    assert cfg.ticks == 300
    assert cfg.seed == 123
    assert cfg.dt == 0.25
    assert cfg.network.background_current == 0.4
    assert cfg.network.spontaneous_spike_rate == 0.2
    assert cfg.network.low_activation_threshold == 5


# ======================================================================
# 合并自 test_cli_experiments.py：单元测试 — 实验管理 CLI 子命令（list / compare / delete）与 run 追踪集成
# ======================================================================


def _seed_db(db_path: Path) -> None:
    """写入两条可对比的实验记录。"""
    with ExperimentStore(db_path) as store:
        store.save(
            ExperimentRecord(
                experiment_id="run_a",
                experiment_name="baseline",
                status="completed",
                created_at="2026-07-31T00:00:01Z",
                seed=42,
                ticks=100,
                neurons=8,
                runtime_seconds=0.5,
                output_dir=str(db_path.parent / "run_a"),
                config={"ticks": 100, "network": {"max_neurons": 50}},
                summary={"total_neurons": 33, "alive_neurons": 33},
                final_metrics={"mean_energy": 0.85},
            )
        )
        store.save(
            ExperimentRecord(
                experiment_id="run_b",
                experiment_name="variant",
                status="completed",
                created_at="2026-07-31T00:00:02Z",
                seed=7,
                ticks=200,
                neurons=16,
                runtime_seconds=1.2,
                output_dir=str(db_path.parent / "run_b"),
                config={"ticks": 200, "network": {"max_neurons": 80}},
                summary={"total_neurons": 50, "alive_neurons": 41},
                final_metrics={"mean_energy": 0.62},
            )
        )


# ── list ────────────────────────────────────────────────────


def test_cmd_list_shows_experiments(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    cli_list.cmd_list(argparse.Namespace(db=str(db), filter=None, limit=0))
    out = capsys.readouterr().out
    assert "共 2 个实验" in out
    assert "run_a" in out
    assert "run_b" in out
    assert "baseline" in out
    assert "33/33" in out


def test_cmd_list_with_filter(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    cli_list.cmd_list(argparse.Namespace(db=str(db), filter="neurons>8", limit=0))
    out = capsys.readouterr().out
    assert "run_b" in out
    assert "run_a" not in out


def test_cmd_list_invalid_filter_reports_error(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    cli_list.cmd_list(argparse.Namespace(db=str(db), filter="neurons", limit=0))
    out = capsys.readouterr().out
    assert "过滤表达式错误" in out


def test_cmd_list_missing_db(tmp_path: Path, capsys) -> None:
    cli_list.cmd_list(argparse.Namespace(db=str(tmp_path / "missing.db"), filter=None, limit=0))
    out = capsys.readouterr().out
    assert "实验数据库不存在" in out


def test_cmd_list_no_match(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    cli_list.cmd_list(argparse.Namespace(db=str(db), filter="seed=999", limit=0))
    out = capsys.readouterr().out
    assert "无匹配实验" in out


# ── compare ─────────────────────────────────────────────────


def test_cmd_compare_outputs_diff_and_metrics(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    cli_compare.cmd_compare(argparse.Namespace(db=str(db), experiment_ids=["run_a", "run_b"]))
    out = capsys.readouterr().out
    assert "实验对比 (2 个实验)" in out
    assert "参数差异" in out
    assert "network.max_neurons" in out
    assert "结果指标对比" in out
    assert "mean_energy" in out
    assert "alive_neurons" in out


def test_cmd_compare_requires_two_ids(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    with pytest.raises(SystemExit) as exc:
        cli_compare.cmd_compare(argparse.Namespace(db=str(db), experiment_ids=["run_a"]))
    assert exc.value.code == 1
    assert "至少需要 2 个实验 ID" in capsys.readouterr().out


def test_cmd_compare_missing_experiment_exits(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    with pytest.raises(SystemExit) as exc:
        cli_compare.cmd_compare(argparse.Namespace(db=str(db), experiment_ids=["run_a", "run_zzz"]))
    assert exc.value.code == 1
    assert "实验不存在: run_zzz" in capsys.readouterr().out


# ── delete ──────────────────────────────────────────────────


def test_cmd_delete_removes_metadata_and_files(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    output_dir = tmp_path / "run_a"
    output_dir.mkdir()
    (output_dir / "run.jsonl").write_text("{}\n", encoding="utf-8")

    cli_delete.cmd_delete(
        argparse.Namespace(db=str(db), experiment_ids=["run_a"], keep_files=False)
    )
    out = capsys.readouterr().out
    assert "已删除: run_a" in out
    assert "含输出文件" in out
    assert not output_dir.exists()
    with ExperimentStore(db) as store:
        assert store.get("run_a") is None
        assert store.get("run_b") is not None


def test_cmd_delete_keep_files(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    output_dir = tmp_path / "run_b"
    output_dir.mkdir()

    cli_delete.cmd_delete(argparse.Namespace(db=str(db), experiment_ids=["run_b"], keep_files=True))
    out = capsys.readouterr().out
    assert "仅元数据" in out
    assert output_dir.exists()


def test_cmd_delete_refuses_mismatched_output_dir(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    unrelated = tmp_path / "unrelated_dir"
    unrelated.mkdir()
    with ExperimentStore(db) as store:
        store.save(ExperimentRecord(experiment_id="run_evil", output_dir=str(unrelated)))
    cli_delete.cmd_delete(
        argparse.Namespace(db=str(db), experiment_ids=["run_evil"], keep_files=False)
    )
    out = capsys.readouterr().out
    assert "拒绝删除" in out
    assert unrelated.exists()  # 目录名不匹配，未被删除
    with ExperimentStore(db) as store:
        assert store.get("run_evil") is None  # 元数据仍被删除


def test_cmd_delete_missing_experiment_skips(tmp_path: Path, capsys) -> None:
    db = tmp_path / "experiments.db"
    _seed_db(db)
    cli_delete.cmd_delete(
        argparse.Namespace(db=str(db), experiment_ids=["run_zzz"], keep_files=False)
    )
    out = capsys.readouterr().out
    assert "实验不存在，跳过: run_zzz" in out
    assert "共删除 0/1" in out


# ── parser 分发 ─────────────────────────────────────────────


def test_main_dispatches_list(monkeypatch) -> None:
    called = {"list": False}
    monkeypatch.setattr(cli_parser, "cmd_list", lambda args: called.update(list=True))
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "list"])
    cli_parser.main()
    assert called["list"] is True


def test_main_dispatches_compare(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_parser, "cmd_compare", lambda args: captured.update(ids=args.experiment_ids)
    )
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "compare", "exp1", "exp2"])
    cli_parser.main()
    assert captured["ids"] == ["exp1", "exp2"]


def test_main_dispatches_delete(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_parser,
        "cmd_delete",
        lambda args: captured.update(ids=args.experiment_ids, keep=args.keep_files),
    )
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "delete", "exp1", "--keep-files"])
    cli_parser.main()
    assert captured["ids"] == ["exp1"]
    assert captured["keep"] is True


# ── run 追踪集成 ────────────────────────────────────────────


def _run_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "ticks": 3,
        "seed": 42,
        "dt": 1.0,
        "neurons": 4,
        "expr": "1+1=2",
        "input_weight": 60.0,
        "stim_interval": 1,
        "output": str(tmp_path),
        "experiment_name": None,
        "config": None,
        "debug": False,
        "log_interval": 1,
        "gzip_output": False,
        "max_file_size": 0,
        "async_log": False,
        "log_buffer_size": 256,
        "visualize": False,
        "plot_interval": 5,
        "checkpoint_interval": 0,
        "checkpoint_dir": str(tmp_path),
        "resume_from": None,
        "background_current": None,
        "spontaneous_spike_rate": None,
        "low_activation_threshold": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_run_tracks_experiment_metadata(tmp_path: Path, capsys) -> None:
    cli_run.cmd_run(_run_args(tmp_path, experiment_name="tracked-exp"))
    capsys.readouterr()

    db = tmp_path / "experiments.db"
    assert db.exists()
    with ExperimentStore(db) as store:
        records = store.list_experiments()
    assert len(records) == 1
    record = records[0]
    assert record.experiment_name == "tracked-exp"
    assert record.status == "completed"
    assert record.seed == 42
    assert record.ticks == 3
    assert record.neurons == 4
    assert record.runtime_seconds is not None
    assert record.finished_at is not None
    assert record.summary.get("total_neurons") is not None
    assert record.final_metrics.get("tick") == 3
    assert record.config.get("ticks") == 3
    assert record.cli_args.get("expr") == "1+1=2"
    # experiment_id 与输出目录名一致
    assert record.output_dir is not None
    assert Path(record.output_dir).name == record.experiment_id


def test_cmd_run_marks_failed_on_exception(tmp_path: Path, monkeypatch, capsys) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(cli_run, "_run_simulation_loop", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cli_run.cmd_run(_run_args(tmp_path))
    capsys.readouterr()

    with ExperimentStore(tmp_path / "experiments.db") as store:
        records = store.list_experiments("status=failed")
    assert len(records) == 1
    assert records[0].error == "simulated crash"


def test_cmd_run_tracking_failure_degrades_gracefully(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_run, "ExperimentStore", lambda *a, **k: (_ for _ in ()).throw(OSError("db locked"))
    )
    cli_run.cmd_run(_run_args(tmp_path))
    out = capsys.readouterr().out
    assert "实验追踪初始化失败" in out
    assert "完成!" in out  # 仿真本身不受影响
