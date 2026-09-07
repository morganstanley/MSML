#!/usr/bin/env python
"""Simple runner script - no PYTHONPATH needed."""
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from alpha_lab.run import run_main
run_main()
