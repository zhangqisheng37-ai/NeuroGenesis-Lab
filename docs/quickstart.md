# 快速开始

本页所有命令均基于当前 CLI 实现（`src/neurogenesis_lab/cli/parser.py`）。

## 1) 安装

```bash
# 基础安装
pip install -e .

# 可视化依赖（可选）
pip install -e ".[viz]"

# Rust 依赖（可选）
pip install -e ".[rust]"
python -m maturin develop --release
```

## 2) 30 秒自检

```bash
python -m neurogenesis_lab --help
python -m neurogenesis_lab run --help
python -m neurogenesis_lab benchmark --help
python -m neurogenesis_lab info
```

`info` 会输出：

- 版本（`v0.6.1`）
- Schema（`2.0.0`）
- Rust Core 可用性与组件列表
- 默认配置 JSON

## 3) 最小可运行命令

```bash
# 默认运行（100 tick）
python -m neurogenesis_lab run

# 覆盖关键参数
python -m neurogenesis_lab run --ticks 200 --seed 123 --neurons 12

# 使用 TOML 配置
python -m neurogenesis_lab run --config config.toml
```

## 4) 长程实验推荐组合

适用于 500+ tick 的持续运行：采样 + 压缩 + 轮转 + 异步写盘。

```bash
python -m neurogenesis_lab run \
  --ticks 2000 \
  --log-interval 10 \
  --gzip-output \
  --max-file-size 5000000 \
  --async-log \
  --log-buffer-size 512
```

## 5) checkpoint 与恢复

```bash
# 每 100 个“本次运行 tick”保存一次 checkpoint
python -m neurogenesis_lab run --ticks 1000 --checkpoint-interval 100 --checkpoint-dir outputs/checkpoints

# 从 checkpoint 恢复后继续运行 200 tick
python -m neurogenesis_lab run --resume-from outputs/checkpoints/checkpoint_tick_1000.pkl --ticks 200
```

## 6) 性能基准

```bash
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100 --rust-size 10000
```

## 7) 输出文件约定

默认输出目录：`outputs/`。

- 主日志：`run.jsonl` 或 `run.jsonl.gz`
- 轮转：`run_0001.jsonl(.gz)`、`run_0002.jsonl(.gz)` …
- checkpoint：`checkpoint_tick_<global_tick>.pkl`

## 8) 实验配置导航（`experiments/`）

- 基础：`experiment_minimal.toml`、`experiment_stable.toml`、`experiment_optimized.toml`
- 对照/专题：`experiment_contrast.toml`、`experiment_spatial.toml`、`experiment_learning.toml`
- 挑战：`challenge_hardcore.toml`、`challenge_logic.toml`、`challenge_war.toml`
- 消融：`ablation_adaptive_on.toml`、`ablation_adaptive_off.toml`

## 9) 常见问题（激活不足）

若运行后 `总放电` / `总传播` 过低，可优先尝试：

- 提高 `--input-weight`（如 60~120）
- 减小 `--stim-interval`（如 1~3）
- 配置中提高 `network.input_current_scale`
- 配置中降低 `neuron.v_th`
- 使用 `--background-current` 或 `--spontaneous-spike-rate`
