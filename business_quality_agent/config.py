# Business Quality Agent — Centralized Path Configuration
#
# All file paths used by the agent are defined here. To override the default
# base folder (~/Desktop/AI_Investment_Committee), set the environment variable:
#
#   export BQA_BASE_DIR="/path/to/your/folder"
#
# On Windows (PowerShell):
#   $env:BQA_BASE_DIR = "C:\Users\you\Documents\AI_Investment_Committee"
#
# On Windows (cmd):
#   set BQA_BASE_DIR=C:\Users\you\Documents\AI_Investment_Committee

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directory — configurable via BQA_BASE_DIR environment variable
# ---------------------------------------------------------------------------

_env_base = os.environ.get("BQA_BASE_DIR")
if _env_base:
    BASE_DIR = Path(_env_base).expanduser().resolve()
else:
    BASE_DIR = Path.home() / "Desktop" / "AI_Investment_Committee"

# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------

CSV_PATH = BASE_DIR / "stock_universe.csv"
OUTPUT_FOLDER = BASE_DIR
CSV_OUTPUT = BASE_DIR / "quality_rankings.csv"
JSON_OUTPUT = BASE_DIR / "agent_handoff.json"
