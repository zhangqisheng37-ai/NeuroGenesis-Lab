"""NeuroGenesis Lab — 统一 CLI 入口

用法:
    python -m neurogenesis_lab run [--ticks N] [--seed S] [--output DIR]
    python -m neurogenesis_lab run --config experiment.toml
    python -m neurogenesis_lab benchmark [--neurons N] [--ticks T]
    python -m neurogenesis_lab info
"""

# --- 向后兼容：保留旧导入路径 ---
from neurogenesis_lab.cli import main  # noqa: F401
from neurogenesis_lab.cli.config_builder import (
    build_sim_config as _build_sim_config,
)  # noqa: F401
from neurogenesis_lab.cli.run import cmd_run  # noqa: F401
from neurogenesis_lab.cli.benchmark import cmd_benchmark  # noqa: F401
from neurogenesis_lab.cli.info import cmd_info  # noqa: F401

if __name__ == "__main__":
    main()
