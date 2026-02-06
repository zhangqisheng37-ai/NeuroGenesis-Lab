import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.learning import MemoryBank, PatternMiner, QueryEngine


def test_pattern_table_answers_known_queries() -> None:
    bank = MemoryBank()
    bank.add_many(["1+1=2", "1+2=3"])

    miner = PatternMiner()
    miner.fit(bank.expressions)
    engine = QueryEngine(miner=miner)

    assert engine.query("1+1=?") == "2"
    assert engine.query("1+2=?") == "3"
    assert engine.query("1+3=?") == "unknown"
