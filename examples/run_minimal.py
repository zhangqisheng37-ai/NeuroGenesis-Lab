import sys
from pathlib import Path
 
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
 
from neurogenesis_lab.config import default_config
from neurogenesis_lab.logger import create_logger, log_tick
 
 
def main() -> None:
     config = default_config()
     logger = create_logger()
     for tick in range(config.ticks):
         log_tick(
             logger,
             tick=tick,
             module="bootstrap",
             message="tick",
             network_size=config.network_size,
             energy_decay=config.energy_decay,
         )
 
 
if __name__ == "__main__":
     main()
