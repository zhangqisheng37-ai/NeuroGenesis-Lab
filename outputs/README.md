# outputs/ 目录约定

本目录存放仿真运行时产物。**除本文件外，整个目录不入版本控制**。
需要长期保留的结论必须导出为图表/汇总数据到 `docs/assets/` 或 `docs/`。

## 唯一元数据真值

`experiments.db`（SQLite）是实验元数据的**唯一真源**，由 `neurogenesis_lab/experiments.py` 维护。
查询请使用 CLI，不要手写 SQL 或另建索引文件：

```bash
python -m neurogenesis_lab list
python -m neurogenesis_lab list --filter "ticks>=500,seed=42"
python -m neurogenesis_lab compare <exp_id_1> <exp_id_2>
```

> 历史提示：早期原型使用 `experiments/index.json` 索引文件，该机制已废弃，
> 旧文件归档在 `archive/legacy_index.json`，仅供格式参考，程序不再读写。

## 目录结构

```text
outputs/
├── README.md              # 本文件（唯一受版本控制的文件）
├── experiments.db         # 实验元数据唯一真值
├── run_<YYYYMMDD>_<HHMMSS>_<ms>/   # 单次运行产物
│   ├── run.jsonl[.gz]     # 逐 tick 采样日志
│   ├── run_0001.jsonl     # 轮转后的分片（启用 --max-file-size 时）
│   └── experiment.json    # 旧原型格式，仅历史目录存在
└── archive/               # 归档区，不参与日常查询
    ├── legacy_index.json  # 已废弃的 JSON 索引
    └── unsorted_runs/     # 无元数据记录的孤儿运行（无 db 条目）
```

## 写入规则

- 新运行由 `run` 命令自动建目录，**不要手工创建或改名**。
- 有意义的实验必须带名字，便于日后检索：

  ```bash
  python -m neurogenesis_lab run --experiment-name learn_on_seed42 --ticks 2000
  ```

- 未命名（`experiment_name` 为空）的运行视同一次性冒烟测试，可随时清理。

## 归档与清理

- 判断孤儿：目录名不在 `list` 输出中的，即为无元数据记录，移到 `archive/unsorted_runs/`。
- 删除前先导出必要指标，`delete` 子命令会同时移除元数据与文件：

  ```bash
  python -m neurogenesis_lab delete <exp_id>            # 元数据 + 文件
  python -m neurogenesis_lab delete <exp_id> --keep-files
  ```

- 安全护栏：`delete` 仅在输出目录名与实验 ID 一致时才删除文件，手工改名的目录不会被误删。
