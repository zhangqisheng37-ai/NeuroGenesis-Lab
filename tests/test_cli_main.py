"""单元测试 — CLI 入口分发与信息输出"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import neurogenesis_lab.__main__ as cli


def test_cmd_info_outputs_version_and_schema(capsys) -> None:
    cli.cmd_info(cli.argparse.Namespace())
    out = capsys.readouterr().out
    assert "NeuroGenesis Lab v" in out
    assert "Schema Version" in out
    assert "默认配置" in out


def test_main_dispatches_info(monkeypatch) -> None:
    called = {"info": False}

    def fake_info(args) -> None:
        called["info"] = True

    monkeypatch.setattr(cli, "cmd_info", fake_info)
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab", "info"])

    cli.main()

    assert called["info"] is True


def test_main_without_command_prints_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["neurogenesis-lab"])

    cli.main()

    out = capsys.readouterr().out
    assert "可用命令" in out
    assert "run" in out
    assert "benchmark" in out
    assert "info" in out
