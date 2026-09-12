"""Compatibility entry point for the Q2-inherited paper builder."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("同步继承模型论文.py")), run_name="__main__")
