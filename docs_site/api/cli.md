# cli — 命令行接口

CLI 子包（v0.6.0 从 `__main__.py` 拆分），提供 `run` / `benchmark` / `info` 三个命令。

## 模块结构

- **`cli.parser`** — argparse 参数定义
- **`cli.config_builder`** — TOML 解析与 SimConfig 构建 (`build_sim_config`)
- **`cli.run`** — `cmd_run()` 仿真运行入口
- **`cli.benchmark`** — `cmd_benchmark()` 性能基准
- **`cli.info`** — `cmd_info()` 版本信息

::: neurogenesis_lab.cli
