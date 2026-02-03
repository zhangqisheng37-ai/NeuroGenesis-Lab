from dataclasses import dataclass
 
 
@dataclass(frozen=True)
class Config:
     ticks: int = 10
     network_size: int = 16
     energy_decay: float = 0.01
     seed: int = 42
 
 
def default_config() -> Config:
     return Config()
