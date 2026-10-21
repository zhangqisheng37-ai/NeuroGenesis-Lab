# cli — 命令行接口

CLI 子包（v0.6.0 从 `__main__.py` 拆分），提供 8 个子命令：
`run` / `serve` / `web` / `list` / `compare` / `delete` / `benchmark` / `info`。

## 模块结构

- **`cli.parser`** — argparse 参数定义
- **`cli.config_builder`** — TOML 解析与 SimConfig 构建 (`build_sim_config`)
- **`cli.run`** — `cmd_run()` 仿真运行入口
- **`cli.serve`** — `cmd_serve()` 独立实时可视化服务
- **`cli.web`** — `cmd_web()` Web 管理界面
- **`cli.list`** — `cmd_list()` 实验列表与过滤查询
- **`cli.compare`** — `cmd_compare()` 多实验参数/指标对比
- **`cli.delete`** — `cmd_delete()` 删除实验元数据与输出
- **`cli.benchmark`** — `cmd_benchmark()` 性能基准
- **`cli.info`** — `cmd_info()` 版本信息

## web

启动 Web 管理界面，提供实验管理、配置编辑、实时监控等功能。

```bash
python -m neurogenesis_lab web [--host HOST] [--port PORT] [--db DB]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | 127.0.0.1 | 绑定主机 |
| `--port` | 8080 | 绑定端口 |
| `--db` | outputs/experiments.db | 实验数据库路径 |

启动后访问 `http://127.0.0.1:8080` 进入仪表盘。

### Web 界面功能

- **仪表盘**：系统概览、最近实验、快速操作入口
- **实验管理**：列表、详情、对比、删除（支持批量操作）
- **配置编辑器**：图形化参数编辑、预设模板、TOML 导出
- **实时监控**：启动仿真并通过 WebSocket 实时查看网络状态

::: neurogenesis_lab.cli
