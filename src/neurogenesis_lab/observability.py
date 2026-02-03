import json
from pathlib import Path
from typing import Iterable


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, records: Iterable[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
