# Business Quality Agent — Stock Universe Reader
# Step 1: Data pipeline validation
#
# Dependency: pandas  (install with: pip install pandas)
#
# Usage:
#   python read_universe.py
#
# Expects the file at:
#   ~/Desktop/AI_Investment_Committee/stock_universe.csv
# Override with: export BQA_BASE_DIR="/your/path"

import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("[ERROR] 'pandas' is not installed.")
    print("        Fix: pip install pandas")
    sys.exit(1)

# Sibling-module import for centralized path config.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from config import CSV_PATH

# Column names to search for, in priority order (case-insensitive).
TICKER_COLUMN_CANDIDATES = [
    "ticker",
    "symbol",
    "tickers",
    "symbols",
    "stock",
    "ticker_symbol",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_ticker_column(df: "pd.DataFrame") -> str:
    """Return the name of the ticker column as it appears in *df*.

    Raises ValueError if no recognised ticker column is found.
    """
    normalised = {col.strip().lower(): col for col in df.columns}

    for candidate in TICKER_COLUMN_CANDIDATES:
        if candidate in normalised:
            return normalised[candidate]

    available = ", ".join(f'"{c}"' for c in df.columns)
    raise ValueError(
        f"Could not detect a ticker column.\n"
        f"  Columns found in the file: {available}\n"
        f"  Expected one of: {', '.join(TICKER_COLUMN_CANDIDATES)}\n"
        f"  Fix: rename the appropriate column in your CSV to 'ticker'."
    )


def load_tickers(path: Path) -> list[str]:
    """Load and return a sorted list of cleaned ticker symbols from *path*."""
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The CSV file was loaded but contains no rows.")

    col = detect_ticker_column(df)

    tickers = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    tickers.sort()
    return tickers


def display_tickers(tickers: list[str], source_file: Path) -> None:
    """Print the ticker list to stdout in a readable format."""
    count = len(tickers)
    separator = "-" * 45

    print(f"\n  Found {count} ticker symbol{'s' if count != 1 else ''} "
          f"in {source_file.name}")
    print(f"  {separator}")
    for i, ticker in enumerate(tickers, start=1):
        print(f"  {i:>4}.  {ticker}")
    print(f"  {separator}")
    print(f"  Pipeline check PASSED — ready for analysis.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Verify the file exists before attempting to load it.
    if not CSV_PATH.exists():
        print(f"\n[ERROR] File not found: {CSV_PATH}")
        print( "        Check that:")
        print( "          1. The file 'stock_universe.csv' is in the folder")
        print(f"             {CSV_PATH.parent}")
        print( "          2. The folder name matches exactly: 'AI_Investment_Committee'")
        print( "          3. The folder is on your Desktop")
        print( "        Or set BQA_BASE_DIR to point to your folder:")
        print( '          export BQA_BASE_DIR="/path/to/your/folder"\n')
        sys.exit(1)

    # 2. Load the tickers.
    try:
        tickers = load_tickers(CSV_PATH)
    except ValueError as exc:
        print(f"\n[ERROR] {exc}\n")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected problem reading the file: {exc}\n")
        sys.exit(1)

    # 3. Guard against an empty ticker list after cleaning.
    if not tickers:
        print("\n[ERROR] The ticker column exists but contains no valid values.")
        print("        Check that the column is not empty or filled with blanks.\n")
        sys.exit(1)

    # 4. Display results.
    display_tickers(tickers, CSV_PATH)


if __name__ == "__main__":
    main()
