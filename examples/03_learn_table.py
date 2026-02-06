import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.learning import MemoryBank, PatternMiner, QueryEngine


def main() -> None:
    bank = MemoryBank()
    bank.add_many(["1+1=2", "1+2=3"])

    miner = PatternMiner()
    miner.fit(bank.expressions)
    engine = QueryEngine(miner=miner)

    for query in ["1+1=?", "1+2=?", "1+3=?"]:
        print({"query": query, "answer": engine.query(query)})


if __name__ == "__main__":
    main()
