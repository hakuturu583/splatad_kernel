"""Make the tests runnable straight from a checkout, installed or not."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # reference_numpy
try:
    import splatad_kernel  # noqa: F401
except ImportError:
    sys.path.insert(0, str(_ROOT / "src"))
