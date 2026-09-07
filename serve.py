#!/usr/bin/env python
"""Simple server script - no PYTHONPATH needed."""
import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

from alpha_lab.server import serve_main
serve_main()
