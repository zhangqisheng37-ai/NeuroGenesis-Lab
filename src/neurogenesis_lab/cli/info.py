"""CLI info 命令 — 版本与配置信息展示"""

from __future__ import annotations

import argparse
import json

from neurogenesis_lab import __version__
from neurogenesis_lab.config import SCHEMA_VERSION, SimConfig
from neurogenesis_lab.core_bridge import rust_core_status


def cmd_info(args: argparse.Namespace) -> None:
    """显示版本与配置信息。"""
    _ = args
    print(f"NeuroGenesis Lab v{__version__}")
    print(f"Schema Version: {SCHEMA_VERSION}")
    cfg = SimConfig()
    rust_status = rust_core_status()
    enabled_components = [
        name for name, enabled in rust_status.get("components", {}).items() if enabled
    ]
    print(f"Rust Core: {'available' if rust_status['available'] else 'unavailable'}")
    if enabled_components:
        print("Rust Components:")
        for comp in enabled_components:
            print(f"  - {comp}")
    print("默认配置:")
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
