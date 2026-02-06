import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder
from neurogenesis_lab.observability import JsonlWriter


def main() -> None:
    encoder = Encoder()
    expr = "1+1=2"
    tokens = encoder.encode_expr(expr)
    events = encoder.to_events(tokens)

    records: list[dict] = []
    for event in events:
        records.append(
            {
                "token_id": event.token_id,
                "t": event.t,
                "src_id": event.src_id,
                "meta": event.meta,
            }
        )
    records.append(
        {
            "type": "summary",
            "expr": expr,
            "token_count": len(tokens),
            "event_count": len(events),
            "unique_tokens": len(set(tokens)),
        }
    )

    output_path = ROOT / "jsonl" / "run_pulse_flow.jsonl"
    JsonlWriter(output_path).write(records)
    print(output_path)


if __name__ == "__main__":
    main()
