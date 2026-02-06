import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.learning import MemoryBank, PatternMiner, QueryEngine
from neurogenesis_lab.observability import JsonlWriter


def main() -> None:
    bank = MemoryBank()
    bank.add_many(["1+1=2", "1+2=3"])

    miner = PatternMiner()
    miner.fit(bank.expressions)
    engine = QueryEngine(miner=miner)

    queries = ["1+1=?", "1+2=?", "1+3=?"]
    records: list[dict] = []
    known = 0
    for query in queries:
        answer = engine.query(query)
        if answer != "unknown":
            known += 1
        records.append({"query": query, "answer": answer})

    records.append(
        {
            "type": "summary",
            "training_samples": len(bank.expressions),
            "queries": len(queries),
            "known": known,
        }
    )

    output_path = ROOT / "jsonl" / "run_learn_table.jsonl"
    JsonlWriter(output_path).write(records)
    print(output_path)


if __name__ == "__main__":
    main()
