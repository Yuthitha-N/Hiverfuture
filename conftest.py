"""Pytest root configuration adding project root to sys.path."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
