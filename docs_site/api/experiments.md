# experiments — 实验管理

实验元数据的 SQLite 存储模块（Phase 8 / P2），为 `run` 命令提供自动实验追踪，并支撑 `list` / `compare` / `delete` 三个 CLI 子命令。

元数据字段与 `outputs/experiments/index.json` 结构对齐：顶层标量（experiment_id、seed、ticks 等）映射为 SQLite 列以支持索引；嵌套结构（config、cli_args、summary、final_metrics）以 JSON 文本列存储。

## 核心接口

- **`ExperimentRecord`** — 单次实验的元数据记录（dataclass），`flatten()` 展平为单层字典用于过滤匹配
- **`ExperimentStore(db_path)`** — SQLite 存储，支持 `save` / `get` / `list_experiments` / `delete` / `export_index`；自动迁移旧原型 schema
- **`parse_filter_expression(expr)`** — 解析过滤表达式（`key op value`，op 支持 `>` `<` `>=` `<=` `=` `!=`，多条件用 `,` 或 ` and ` 连接）
- **`matches_filters(record, conditions)`** — 判断记录是否满足全部过滤条件（AND 语义）

## CLI 用法

```bash
# 运行时自动追踪（可选命名）
python -m neurogenesis_lab run --ticks 100 --experiment-name my-exp

# 列出实验（支持过滤）
python -m neurogenesis_lab list --filter "neurons>4,seed=42"

# 对比实验（参数差异 + 结果指标）
python -m neurogenesis_lab compare run_xxx run_yyy

# 删除实验（元数据 + 输出文件；--keep-files 仅删元数据）
python -m neurogenesis_lab delete run_xxx
```

::: neurogenesis_lab.experiments
