# itog_module.py
# Thin wrapper to make the Cyrillic Итог.py importable as "itog_module"
import importlib.util
import sys
from pathlib import Path

base = Path(__file__).parent
itog_path = base / "Итог.py"
if not itog_path.exists():
    raise ImportError(f"Cannot find Итог.py next to {__file__}")

spec = importlib.util.spec_from_file_location("itog_module", str(itog_path))
module = importlib.util.module_from_spec(spec)
# register the module name so child processes can import it
sys.modules["itog_module"] = module
spec.loader.exec_module(module)
# no further action required; the module is now importable as "itog_module"