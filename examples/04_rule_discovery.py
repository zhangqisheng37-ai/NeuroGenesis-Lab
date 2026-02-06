import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from neurogenesis_lab.encoding import Tokenizer
from neurogenesis_lab.learning import RuleComposer, RuleDiscovery


def main() -> None:
    tokenizer = Tokenizer()
    samples = [
        tokenizer.encode("1+1=2"),
        tokenizer.encode("1+2=3"),
        tokenizer.encode("2+1=3"),
    ]
    discovery = RuleDiscovery(min_support=2)
    store = discovery.discover(samples)
    composer = RuleComposer(store=store)

    query = tokenizer.encode("1+3=?")
    missing_token = tokenizer.encode("?")[0]
    masked = [token if token != missing_token else -1 for token in query]
    filled = composer.apply(masked, missing_token=-1)
    answer = "".join(tokenizer.decode_token(token) for token in filled)
    print({"query": "1+3=?", "filled": answer})


if __name__ == "__main__":
    main()
