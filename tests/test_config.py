"""单元测试 — config 模块"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

import pytest
from neurogenesis_lab.config import (
    Config,
    EvolutionParams,
    NetworkParams,
    NeuronParams,
    SimConfig,
    default_config,
    default_sim_config,
    seed_all,
    SCHEMA_VERSION,
)


class TestConfig:
    def test_default_config(self) -> None:
        cfg = default_config()
        assert cfg.ticks == 10
        assert cfg.seed == 42

    def test_config_frozen(self) -> None:
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.ticks = 100  # type: ignore


class TestNeuronParams:
    def test_valid(self) -> None:
        params = NeuronParams()
        errors = params.validate()
        assert errors == [], f"默认参数不应有错误: {errors}"

    def test_invalid_tau_m(self) -> None:
        params = NeuronParams(tau_m=-1.0)
        errors = params.validate()
        assert any("tau_m" in e for e in errors)

    def test_invalid_nan(self) -> None:
        params = NeuronParams(v_rest=float("nan"))
        errors = params.validate()
        assert any("v_rest" in e for e in errors)


class TestNetworkParams:
    def test_valid(self) -> None:
        params = NetworkParams()
        errors = params.validate()
        assert errors == []

    def test_invalid_share(self) -> None:
        params = NetworkParams(reproduction_share=1.5)
        errors = params.validate()
        assert any("reproduction_share" in e for e in errors)

    def test_invalid_negative_decay(self) -> None:
        params = NetworkParams(energy_decay=-0.1)
        errors = params.validate()
        assert any("energy_decay" in e for e in errors)


class TestEvolutionParams:
    def test_valid(self) -> None:
        params = EvolutionParams()
        errors = params.validate()
        assert errors == []

    def test_invalid_amplitude(self) -> None:
        params = EvolutionParams(mutation_amplitude=-0.1)
        errors = params.validate()
        assert any("mutation_amplitude" in e for e in errors)


class TestSimConfig:
    def test_default_sim_config(self) -> None:
        cfg = default_sim_config()
        assert cfg.ticks == 100
        assert cfg.seed == 42
        assert cfg.schema_version == SCHEMA_VERSION

    def test_validate_all_good(self) -> None:
        cfg = SimConfig()
        errors = cfg.validate()
        assert errors == [], f"默认 SimConfig 不应有错误: {errors}"

    def test_validate_invalid_ticks(self) -> None:
        cfg = SimConfig(ticks=-1)
        errors = cfg.validate()
        assert any("ticks" in e for e in errors)

    def test_validate_invalid_dt(self) -> None:
        cfg = SimConfig(dt=0.0)
        errors = cfg.validate()
        assert any("dt" in e for e in errors)

    def test_to_dict(self) -> None:
        cfg = SimConfig()
        d = cfg.to_dict()
        assert d["schema_version"] == SCHEMA_VERSION
        assert d["seed"] == 42
        assert "neuron" in d
        assert "network" in d
        assert "evolution" in d


class TestSeedAll:
    def test_seed_reproducibility(self) -> None:
        import random
        seed_all(123)
        a = random.random()
        seed_all(123)
        b = random.random()
        assert a == b, "同一 seed 应产生相同随机数"
