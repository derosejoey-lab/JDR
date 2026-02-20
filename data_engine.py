"""
data_engine.py
==============
Data procurement and feature engineering layer for the Market Regime
Detection application.

Public API
----------
get_market_data(ticker, start_date, end_date=None)
    Downloads prices, fetches FRED macro series, engineers all features,
    and returns (feature_df, scaled_matrix, scaler).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CREDIT_TICKERS: list[str] = ["HYG", "LQD"]
FRED_SERIES: dict[str, str] = {
    "UNRATE": "Unemployment_Rate",
    "FEDFUNDS": "Fed_Funds_Rate",
}
VOL_WINDOWS: list[int] = [21, 63]  # trading-day look-backs for realized vol


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily Adj Close prices via yfinance for one or more tickers.

    Returns a DataFrame with a DatetimeIndex and one column per ticker.
    Raises ValueError if no data is returned for any ticker.
    """
    logger.info("Downloading price data for: %s", tickers)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    # yfinance returns multi-level columns when >1 ticker is requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"].copy()
    else:
        # Single ticker — column name is the metric, not the ticker symbol
        prices = raw[["Adj Close"]].copy()
        prices.columns = tickers

    # Ensure DatetimeIndex with timezone stripped
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    missing = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise ValueError(f"No price data returned for ticker(s): {missing}")

    logger.info("Price data shape: %s", prices.shape)
    return prices


def _fetch_fred_series(series_ids: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch monthly macro series from FRED via pandas_datareader.

    Returns a DataFrame indexed by date with columns named by series ID.
    """
    logger.info("Fetching FRED series: %s", series_ids)
    frames: list[pd.DataFrame] = []
    for sid in series_ids:
        try:
            s = pdr.DataReader(sid, "fred", start=start, end=end)
            s.columns = [sid]
            frames.append(s)
        except Exception as exc:  # network or FRED outage
            logger.warning("Could not fetch FRED series '%s': %s", sid, exc)

    if not frames:
        raise RuntimeError("Failed to retrieve any FRED series. Check network/FRED availability.")

    macro = pd.concat(frames, axis=1)
    macro.index = pd.to_datetime(macro.index).tz_localize(None)
    logger.info("FRED macro shape (before reindex): %s", macro.shape)
    return macro


def _align_macro_to_daily(macro: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Reindex monthly FRED observations onto a daily business-day index using
    forward-fill.  This ensures every trading day carries the most recent
    available macro reading.
    """
    macro_daily = macro.reindex(daily_index, method="ffill")
    return macro_daily


def _compute_log_returns(prices: pd.Series, name: str) -> pd.Series:
    """Compute log-returns: ln(P_t / P_{t-1})."""
    lr = np.log(prices / prices.shift(1))
    lr.name = name
    return lr


def _compute_realized_vol(log_returns: pd.Series, windows: list[int]) -> pd.DataFrame:
    """
    Rolling realized volatility for each window (std of log-returns).
    Column names: Vol_{w}d.
    """
    vols = {}
    for w in windows:
        vols[f"Vol_{w}d"] = log_returns.rolling(window=w, min_periods=w).std()
    return pd.DataFrame(vols, index=log_returns.index)


def _compute_credit_stress(hyg: pd.Series, lqd: pd.Series) -> pd.Series:
    """
    Credit Stress Proxy = log(HYG) - log(LQD).

    Captures the spread between high-yield and investment-grade bond prices.
    A falling value signals rising credit stress (HYG cheapens vs LQD).
    We take the first difference so the feature is stationary.
    """
    spread_level = np.log(hyg) - np.log(lqd)
    stress = spread_level.diff()          # first-difference → stationary
    stress.name = "Credit_Stress"
    return stress


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_market_data(
    ticker: str,
    start_date: str,
    end_date: Optional[str] = None,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    """
    Orchestrate data procurement and feature engineering.

    Parameters
    ----------
    ticker : str
        Primary equity/ETF ticker (e.g. 'SPY').
    start_date : str
        ISO-format start date, e.g. '2005-01-01'.
    end_date : str, optional
        ISO-format end date.  Defaults to today.

    Returns
    -------
    feature_df : pd.DataFrame
        Clean DataFrame with DatetimeIndex containing all engineered features.
        Columns:
            Log_Return        – daily log-return of `ticker`
            Vol_21d           – 21-day rolling realized volatility
            Vol_63d           – 63-day rolling realized volatility
            Credit_Stress     – daily change in log(HYG) − log(LQD) spread
            Unemployment_Rate – UNRATE (forward-filled to daily)
            Fed_Funds_Rate    – FEDFUNDS (forward-filled to daily)
    scaled_matrix : np.ndarray, shape (n_obs, n_features)
        StandardScaler-normalized version of feature_df.values.
    scaler : StandardScaler
        Fitted scaler instance (needed for inverse-transform in model layer).

    Raises
    ------
    ValueError
        If the ticker or credit proxy tickers return no data.
    RuntimeError
        If FRED data cannot be retrieved.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # 1. Price data
    # ------------------------------------------------------------------
    all_tickers = list(dict.fromkeys([ticker] + CREDIT_TICKERS))  # deduplicate
    prices = _download_prices(all_tickers, start=start_date, end=end_date)

    # ------------------------------------------------------------------
    # 2. FRED macro data → forward-fill to daily index
    # ------------------------------------------------------------------
    macro_raw = _fetch_fred_series(list(FRED_SERIES.keys()), start=start_date, end=end_date)
    macro = _align_macro_to_daily(macro_raw, daily_index=prices.index)
    macro.rename(columns=FRED_SERIES, inplace=True)

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    log_ret = _compute_log_returns(prices[ticker], name="Log_Return")
    vols = _compute_realized_vol(log_ret, windows=VOL_WINDOWS)
    credit_stress = _compute_credit_stress(prices["HYG"], prices["LQD"])

    # ------------------------------------------------------------------
    # 4. Assemble feature matrix
    # ------------------------------------------------------------------
    feature_df = pd.concat(
        [log_ret, vols, credit_stress, macro],
        axis=1,
    )

    # Drop rows with any NaN (artifacts of rolling windows and ffill boundaries)
    n_before = len(feature_df)
    feature_df.dropna(inplace=True)
    n_dropped = n_before - len(feature_df)
    logger.info(
        "Feature matrix assembled: %d rows (%d dropped for NaN).",
        len(feature_df),
        n_dropped,
    )

    if feature_df.empty:
        raise ValueError(
            "Feature matrix is empty after NaN removal. "
            "Try extending start_date to allow rolling windows to warm up."
        )

    # ------------------------------------------------------------------
    # 5. Standard scaling
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    scaled_matrix: np.ndarray = scaler.fit_transform(feature_df.values)

    logger.info(
        "Final feature columns: %s",
        list(feature_df.columns),
    )

    return feature_df, scaled_matrix, scaler


# ---------------------------------------------------------------------------
# Quick self-test (run: python data_engine.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    TICKER = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    START = sys.argv[2] if len(sys.argv) > 2 else "2010-01-01"

    print(f"\nRunning data_engine smoke test  →  ticker={TICKER}, start={START}\n")
    df, matrix, scaler_ = get_market_data(TICKER, START)

    print("=" * 60)
    print(f"feature_df shape : {df.shape}")
    print(f"scaled_matrix shape: {matrix.shape}")
    print(f"Date range       : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Columns          : {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDescriptive stats:")
    print(df.describe().round(6))
    print("\nScaled matrix — mean (should be ~0):", matrix.mean(axis=0).round(4))
    print("Scaled matrix — std  (should be ~1):", matrix.std(axis=0).round(4))
