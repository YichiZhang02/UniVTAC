"""Compatibility entry point for the original ResNet encoder training."""
import runpy
import sys
from pathlib import Path

folder = Path(__file__).resolve().parent / 'ResNet'
sys.path.insert(0, str(folder))
runpy.run_path(str(folder / 'legacy_train.py'), run_name='__main__')
