#!/usr/bin/env python3
"""
Tau-Helper standalone runner.

This script can be called from the repository root:
    python tau_helper/run.py info
    python tau_helper/run.py evaluate "instruction"

All code remains within the tau_helper directory.
"""

import sys
import os
from pathlib import Path

# Get the directory where this script is located (tau_helper/)
SCRIPT_DIR = Path(__file__).parent.absolute()

# Get the parent directory (repository root)
REPO_ROOT = SCRIPT_DIR.parent

# Add repo root to Python path so we can import tau_helper
sys.path.insert(0, str(REPO_ROOT))

# Note: We do NOT change directory. The tool needs CWD to be the repo root
# to find domains/. The .env file is loaded relative to the script location
# in llm.py using Path(__file__).parent, so it doesn't need CWD to be tau_helper/

# Now import and run the CLI
from tau_helper.cli import cli

if __name__ == "__main__":
    cli()

