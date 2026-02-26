# Business Quality Agent — Environment & Path Diagnostic
#
# Run this script FIRST to verify that your system is set up correctly.
#
# Usage:
#   python business_quality_agent/check_paths.py
#
# It checks:
#   1. Required Python packages (pandas, anthropic, tavily-python, pypdf)
#   2. Base directory exists (~/Desktop/AI_Investment_Committee or BQA_BASE_DIR)
#   3. stock_universe.csv exists and is readable
#   4. PDF files (optional) are present
#   5. API keys are set (ANTHROPIC_API_KEY, TAVILY_API_KEY)

import os
import sys
from pathlib import Path

# Use the centralized config for all paths.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from config import BASE_DIR, CSV_PATH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass = 0
_warn = 0
_fail = 0


def ok(msg: str) -> None:
    global _pass
    _pass += 1
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    global _warn
    _warn += 1
    print(f"  [WARN] {msg}")


def fail(msg: str) -> None:
    global _fail
    _fail += 1
    print(f"  [FAIL] {msg}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_packages() -> None:
    print("\n--- Python Packages ---")

    for pkg, import_name, required in [
        ("pandas", "pandas", True),
        ("anthropic", "anthropic", True),
        ("tavily-python", "tavily", True),
        ("pypdf", "pypdf", False),
    ]:
        try:
            __import__(import_name)
            ok(f"{pkg} is installed")
        except ImportError:
            if required:
                fail(f"{pkg} is NOT installed.  Fix: pip install {pkg}")
            else:
                warn(f"{pkg} is NOT installed (optional).  Fix: pip install {pkg}")


def check_base_dir() -> None:
    print("\n--- Base Directory ---")

    env_val = os.environ.get("BQA_BASE_DIR")
    if env_val:
        ok(f"BQA_BASE_DIR is set: {env_val}")
    else:
        print(f"  [INFO] BQA_BASE_DIR is not set — using default: {BASE_DIR}")
        print(f"         To override, set: export BQA_BASE_DIR=\"/your/path\"")

    if BASE_DIR.exists():
        ok(f"Base directory exists: {BASE_DIR}")
    else:
        fail(f"Base directory NOT found: {BASE_DIR}")
        print(f"         Create it with: mkdir -p \"{BASE_DIR}\"")
        if sys.platform == "win32":
            win_path = str(BASE_DIR).replace("/", "\\")
            print(f"         On Windows:     mkdir \"{win_path}\"")


def check_csv() -> None:
    print("\n--- Stock Universe CSV ---")

    if CSV_PATH.exists():
        ok(f"Found: {CSV_PATH}")
        size = CSV_PATH.stat().st_size
        if size == 0:
            fail("File is empty (0 bytes)")
        else:
            ok(f"File size: {size:,} bytes")

            # Quick check: try reading the header
            try:
                with open(CSV_PATH, "r", encoding="utf-8") as f:
                    header = f.readline().strip()
                if header:
                    ok(f"Header: {header}")
                else:
                    warn("Could not read header line")
            except Exception as exc:
                warn(f"Could not read file: {exc}")
    else:
        fail(f"NOT found: {CSV_PATH}")
        print(f"         Place your stock_universe.csv file in: {BASE_DIR}")


def check_pdfs() -> None:
    print("\n--- PDF Files (optional) ---")

    if not BASE_DIR.exists():
        warn(f"Base directory missing — cannot check for PDFs")
        return

    pdfs = list(BASE_DIR.glob("*.pdf")) + list(BASE_DIR.glob("*.PDF"))
    # Deduplicate (case-insensitive filesystems)
    seen = set()
    unique = []
    for p in pdfs:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    if unique:
        ok(f"Found {len(unique)} PDF file(s):")
        for p in unique:
            print(f"           - {p.name}")
    else:
        warn("No PDF files found — analysis will use hardcoded principles only")
        print(f"         Place quality-investing PDFs in: {BASE_DIR}")


def check_api_keys() -> None:
    print("\n--- API Keys ---")

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        masked = key[:10] + "..." + key[-4:] if len(key) > 14 else "***"
        ok(f"ANTHROPIC_API_KEY is set ({masked})")
    else:
        fail("ANTHROPIC_API_KEY is NOT set")
        print('         Fix: export ANTHROPIC_API_KEY="sk-ant-..."')

    key = os.environ.get("TAVILY_API_KEY", "")
    if key:
        masked = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
        ok(f"TAVILY_API_KEY is set ({masked})")
    else:
        fail("TAVILY_API_KEY is NOT set")
        print('         Fix: export TAVILY_API_KEY="tvly-..."')
        print("         Get a free key at: https://tavily.com")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    border = "=" * 60
    print(f"\n{border}")
    print("  BUSINESS QUALITY AGENT — ENVIRONMENT CHECK")
    print(f"{border}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  Platform:  {sys.platform}")
    print(f"  Home:      {Path.home()}")
    print(f"  Base dir:  {BASE_DIR}")

    check_packages()
    check_base_dir()
    check_csv()
    check_pdfs()
    check_api_keys()

    print(f"\n{border}")
    print(f"  Results:  {_pass} passed  |  {_warn} warnings  |  {_fail} failed")
    print(f"{border}")

    if _fail > 0:
        print("\n  Fix the [FAIL] items above before running the agent.\n")
        sys.exit(1)
    elif _warn > 0:
        print("\n  All required checks passed. Warnings are optional.\n")
    else:
        print("\n  All checks passed — ready to run!\n")


if __name__ == "__main__":
    main()
