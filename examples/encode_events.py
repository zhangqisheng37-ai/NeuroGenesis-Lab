import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Encoder


def main() -> None:
    encoder = Encoder()
    expr = "1+1=2"
    tokens = encoder.encode_expr(expr)
    events = encoder.to_events(tokens)
    for event in events:
        print(asdict(event))


if __name__ == "__main__":
    main()
