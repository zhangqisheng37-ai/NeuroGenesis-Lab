import json
import logging
from typing import Any
 
 
def create_logger(name: str = "neurogenesis") -> logging.Logger:
     logger = logging.getLogger(name)
     if logger.handlers:
         return logger
     logger.setLevel(logging.INFO)
     handler = logging.StreamHandler()
     formatter = logging.Formatter("%(message)s")
     handler.setFormatter(formatter)
     logger.addHandler(handler)
     return logger
 
 
def log_tick(
     logger: logging.Logger,
     tick: int,
     module: str,
     message: str,
     **extra: Any,
 ) -> None:
     payload = {"tick": tick, "module": module, "message": message, **extra}
     logger.info(json.dumps(payload, ensure_ascii=False))
