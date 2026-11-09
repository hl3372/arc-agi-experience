"""Put this folder on sys.path so ``registry``, ``prompts``, ``generators`` and ``common`` import from any cwd."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
