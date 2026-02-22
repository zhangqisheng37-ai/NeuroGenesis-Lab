# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 语言要求

所有回答、汇报、代码注释和文档必须使用中文（简体）。

## Project Overview

NeuroGenesis Lab is a Python-based neural evolution simulation framework for studying digital life emergence, learning, and neural network dynamics. It combines Spiking Neural Networks (SNN) using Leaky Integrate-and-Fire (LIF) neurons, evolutionary algorithms (natural selection, reproduction, mutation, death, scavenger-based corpse recycling), and a hybrid learning engine (symbolic rule discovery + neural dynamics prediction).

## Common Commands

```bash
# Install in development mode
pip install -e .

# (Optional) Install Rust build deps + activate Rust extension
pip install -e .[rust]
python -m maturin develop --release

# Run simulation (default 100 ticks)
python -m neurogenesis_lab run

# Run with custom parameters
python -m neurogenesis_lab run --ticks 200 --seed 123 --neurons 12 --config config.toml

# Run with output governance (sampling + gzip + rotation)
python -m neurogenesis_lab run --log-interval 10 --gzip-output --max-file-size 5000000

# Run with async logging (double buffering)
python -m neurogenesis_lab run --async-log --log-buffer-size 512

# Performance benchmark
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100

# Benchmark with Rust micro benchmark size
python -m neurogenesis_lab benchmark --neurons 5000 --ticks 100 --rust-size 10000

# View version and default config
python -m neurogenesis_lab info

# Run tests (75% minimum coverage required)
pytest

# Run a specific test
pytest tests/test_network.py

# Lint code
ruff check src/
```

## Architecture

The project uses a modular architecture under `src/neurogenesis_lab/`:

- **`config.py`**: Central configuration system with `SimConfig`, `NeuronParams`, `NetworkParams`, `EvolutionParams`
- **`neuron.py`**: Neuron models - `LIFNeuron` (primary), `IzhikevichNeuron`, `AdExNeuron`
- **`network.py`**: Network engine with event queue, evolution logic, and spatial grid indexing
- **`encoding.py`**: Token and Event encoding system for spike representation
- **`learning.py`**: Learning engine with `RuleDiscovery`, `PatternMiner`, `QueryEngine` for symbolic rule extraction
- **`observability.py`**: Metrics, logging, and experiment management
- **`vectorized.py`**: High-performance vectorized simulation kernel using NumPy + scipy.sparse for large-scale networks (10k+ neurons)
- **`protocols.py`**: Unified engine interface — `TickSnapshot`, `EngineSummary`, `SimEngine` Protocol
- **`logger.py`**: Logging utilities

Two network implementations exist: the standard `Network` class and `VectorizedNetwork` for performance-critical simulations. Both conform to the `SimEngine` protocol defined in `protocols.py`, returning unified `TickSnapshot` and `EngineSummary` types.

## Configuration

- Default config: `config.toml`
- Experiment configs: `experiments/*.toml` (e.g., `experiment_evolution.toml`, `experiment_optimized.toml`)
- Override via CLI: `--config`, `--ticks`, `--neurons`, `--seed`, etc.

## Testing

Tests use pytest with coverage. Test files in `tests/` follow naming convention `test_*.py` or `XX_*.py` for numbered integration tests.
