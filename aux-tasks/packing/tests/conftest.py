import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pv2.paths import ensure_generator_package  # noqa: E402

ensure_generator_package()
