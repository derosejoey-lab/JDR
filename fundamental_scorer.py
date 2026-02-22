#!/usr/bin/env python3
"""
Fundamental Stock Scoring Tool
================================
This tool reads a list of stock tickers, pulls financial data from Yahoo Finance,
scores each stock on fundamental metrics, and produces a ranked "Fundamental Rating"
from 0 to 100.

Weights (set by user):
  - ROIC & Capital Efficiency: 20%
  - Profitability: 20%
  - Balance Sheet Strength: 15%
  - Growth: 10%
  - Valuation: 35%

Inspired by: Joel Greenblatt, Philip Fisher, Peter Lynch, Chris Hohn, Stanley Druckenmiller
"""

import pandas as pd
import numpy as np
import yfinance as yf
import csv
import sys
import time
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION - Weights for each category (must sum to 1.0)
# ============================================================
CATEGORY_WEIGHTS = {
    'roic_efficiency': 0.20,   # ROIC & Capital Efficiency
    'profitability':   0.20,   # Profitability margins
    'balance_sheet':   0.15,   # Balance Sheet Strength
    'growth':          0.10,   # Revenue & Earnings Growth
    'valuation':       0.35,   # Valuation (Greenblatt/Druckenmiller lens)
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_get(d, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d


def percentile_score(value, values_series, higher_is_better=True):
    """
    Score a value from 1-10 based on where it falls relative to all other values.
    If higher_is_better=True, a value in the top 10% gets a 10.
    If higher_is_better=False (like debt ratio), a value in the bottom 10% gets a 10.
    """
    if value is None or pd.isna(value):
        return None
    clean = values_series.dropna()
    if len(clean) == 0:
        return None
    rank = (clean < value).sum() / len(clean)
    if not higher_is_better:
        rank = 1 - rank
    return max(1, min(10, round(rank * 9 + 1)))


def fetch_stock_data(ticker_symbol):
    """
    Pull all the financial data we need for one stock from Yahoo Finance.
    Returns a dictionary of raw metrics or None if data can't be retrieved.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        if not info or info.get('quoteType') is None:
            return None

        # Get financial statements
        try:
            income_stmt = stock.income_stmt
        except:
            income_stmt = pd.DataFrame()
        try:
            balance_sheet = stock.balance_sheet
        except:
            balance_sheet = pd.DataFrame()
        try:
            cash_flow = stock.cashflow
        except:
            cash_flow = pd.DataFrame()

        metrics = {}
        metrics['ticker'] = ticker_symbol
        metrics['name'] = info.get('shortName', info.get('longName', ticker_symbol))
        metrics['sector'] = info.get('sector', 'N/A')
        metrics['industry'] = info.get('industry', 'N/A')
        metrics['market_cap'] = info.get('marketCap', None)

        # ----------------------------------------------------------
        # ROIC & CAPITAL EFFICIENCY METRICS
        # ----------------------------------------------------------

        # Current ROIC = EBIT * (1 - tax rate) / Invested Capital
        # Invested Capital = Total Equity + Total Debt - Cash
        try:
            if not income_stmt.empty and not balance_sheet.empty:
                ebit = None
                for label in ['EBIT', 'Operating Income']:
                    if label in income_stmt.index:
                        ebit = income_stmt.loc[label].iloc[0]
                        break

                total_equity = None
                for label in ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']:
                    if label in balance_sheet.index:
                        total_equity = balance_sheet.loc[label].iloc[0]
                        break

                total_debt = None
                for label in ['Total Debt', 'Long Term Debt', 'Long Term Debt And Capital Lease Obligation']:
                    if label in balance_sheet.index:
                        total_debt = balance_sheet.loc[label].iloc[0]
                        break
                if total_debt is None:
                    total_debt = 0

                cash = None
                for label in ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']:
                    if label in balance_sheet.index:
                        cash = balance_sheet.loc[label].iloc[0]
                        break
                if cash is None:
                    cash = 0

                tax_rate = info.get('effectiveTaxRate', None)
                if tax_rate is None:
                    # Try to calculate from financial statements
                    if 'Tax Provision' in income_stmt.index and 'Pretax Income' in income_stmt.index:
                        pretax = income_stmt.loc['Pretax Income'].iloc[0]
                        tax_prov = income_stmt.loc['Tax Provision'].iloc[0]
                        if pretax and pretax > 0:
                            tax_rate = tax_prov / pretax
                if tax_rate is None:
                    tax_rate = 0.21  # Default US corporate rate

                if ebit is not None and total_equity is not None:
                    nopat = ebit * (1 - tax_rate)
                    invested_capital = total_equity + total_debt - cash
                    if invested_capital > 0:
                        metrics['roic'] = (nopat / invested_capital) * 100
                    else:
                        metrics['roic'] = None
                else:
                    metrics['roic'] = None

                # 5-Year Average ROIC & Trend
                roic_history = []
                years_available = min(len(income_stmt.columns), len(balance_sheet.columns), 4)
                for i in range(years_available):
                    try:
                        yr_ebit = None
                        for label in ['EBIT', 'Operating Income']:
                            if label in income_stmt.index:
                                yr_ebit = income_stmt.loc[label].iloc[i]
                                break
                        yr_equity = None
                        for label in ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']:
                            if label in balance_sheet.index:
                                yr_equity = balance_sheet.loc[label].iloc[i]
                                break
                        yr_debt = None
                        for label in ['Total Debt', 'Long Term Debt', 'Long Term Debt And Capital Lease Obligation']:
                            if label in balance_sheet.index:
                                yr_debt = balance_sheet.loc[label].iloc[i]
                                break
                        if yr_debt is None:
                            yr_debt = 0
                        yr_cash = None
                        for label in ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']:
                            if label in balance_sheet.index:
                                yr_cash = balance_sheet.loc[label].iloc[i]
                                break
                        if yr_cash is None:
                            yr_cash = 0

                        if yr_ebit is not None and yr_equity is not None:
                            yr_nopat = yr_ebit * (1 - tax_rate)
                            yr_ic = yr_equity + yr_debt - yr_cash
                            if yr_ic > 0:
                                roic_history.append((yr_nopat / yr_ic) * 100)
                    except:
                        pass

                if len(roic_history) >= 2:
                    metrics['roic_5yr_avg'] = np.mean(roic_history)
                    # Trend: compare most recent to oldest (positive = improving)
                    metrics['roic_trend'] = roic_history[0] - roic_history[-1]
                else:
                    metrics['roic_5yr_avg'] = metrics.get('roic', None)
                    metrics['roic_trend'] = None

                # ROIIC (Return on Incremental Invested Capital)
                # Change in NOPAT / Change in Invested Capital over the period
                if len(roic_history) >= 2 and years_available >= 2:
                    try:
                        # Most recent NOPAT
                        recent_ebit = None
                        oldest_ebit = None
                        for label in ['EBIT', 'Operating Income']:
                            if label in income_stmt.index:
                                recent_ebit = income_stmt.loc[label].iloc[0]
                                oldest_ebit = income_stmt.loc[label].iloc[min(years_available-1, len(income_stmt.columns)-1)]
                                break

                        recent_equity = oldest_equity = None
                        for label in ['Stockholders Equity', 'Total Stockholder Equity', 'Common Stock Equity']:
                            if label in balance_sheet.index:
                                recent_equity = balance_sheet.loc[label].iloc[0]
                                oldest_equity = balance_sheet.loc[label].iloc[min(years_available-1, len(balance_sheet.columns)-1)]
                                break

                        recent_debt = oldest_debt = 0
                        for label in ['Total Debt', 'Long Term Debt', 'Long Term Debt And Capital Lease Obligation']:
                            if label in balance_sheet.index:
                                recent_debt = balance_sheet.loc[label].iloc[0] or 0
                                oldest_debt = balance_sheet.loc[label].iloc[min(years_available-1, len(balance_sheet.columns)-1)] or 0
                                break

                        recent_cash = oldest_cash = 0
                        for label in ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments']:
                            if label in balance_sheet.index:
                                recent_cash = balance_sheet.loc[label].iloc[0] or 0
                                oldest_cash = balance_sheet.loc[label].iloc[min(years_available-1, len(balance_sheet.columns)-1)] or 0
                                break

                        if recent_ebit is not None and oldest_ebit is not None and recent_equity is not None and oldest_equity is not None:
                            delta_nopat = (recent_ebit * (1-tax_rate)) - (oldest_ebit * (1-tax_rate))
                            delta_ic = (recent_equity + recent_debt - recent_cash) - (oldest_equity + oldest_debt - oldest_cash)
                            if delta_ic != 0 and abs(delta_ic) > 1000:
                                metrics['roiic'] = (delta_nopat / delta_ic) * 100
                            else:
                                metrics['roiic'] = None
                        else:
                            metrics['roiic'] = None
                    except:
                        metrics['roiic'] = None
                else:
                    metrics['roiic'] = None

                # Reinvestment Rate = (CapEx + Change in Working Capital) / NOPAT
                try:
                    if not cash_flow.empty and ebit is not None:
                        capex = None
                        for label in ['Capital Expenditure', 'Capital Expenditures']:
                            if label in cash_flow.index:
                                capex = abs(cash_flow.loc[label].iloc[0])
                                break
                        nopat_val = ebit * (1 - tax_rate)
                        if capex is not None and nopat_val > 0:
                            metrics['reinvestment_rate'] = (capex / nopat_val) * 100
                        else:
                            metrics['reinvestment_rate'] = None
                    else:
                        metrics['reinvestment_rate'] = None
                except:
                    metrics['reinvestment_rate'] = None
            else:
                metrics['roic'] = None
                metrics['roic_5yr_avg'] = None
                metrics['roic_trend'] = None
                metrics['roiic'] = None
                metrics['reinvestment_rate'] = None
        except Exception as e:
            metrics['roic'] = None
            metrics['roic_5yr_avg'] = None
            metrics['roic_trend'] = None
            metrics['roiic'] = None
            metrics['reinvestment_rate'] = None

        # ----------------------------------------------------------
        # PROFITABILITY METRICS
        # ----------------------------------------------------------
        try:
            metrics['gross_margin'] = info.get('grossMargins', None)
            if metrics['gross_margin'] is not None:
                metrics['gross_margin'] = metrics['gross_margin'] * 100

            metrics['operating_margin'] = info.get('operatingMargins', None)
            if metrics['operating_margin'] is not None:
                metrics['operating_margin'] = metrics['operating_margin'] * 100

            metrics['profit_margin'] = info.get('profitMargins', None)
            if metrics['profit_margin'] is not None:
                metrics['profit_margin'] = metrics['profit_margin'] * 100

            # Free Cash Flow Margin
            fcf = info.get('freeCashflow', None)
            revenue = info.get('totalRevenue', None)
            if fcf is not None and revenue is not None and revenue > 0:
                metrics['fcf_margin'] = (fcf / revenue) * 100
            else:
                metrics['fcf_margin'] = None

            # SGA % of Revenue (Philip Fisher metric)
            try:
                if not income_stmt.empty:
                    sga = None
                    for label in ['Selling General And Administration', 'Selling And Marketing Expense']:
                        if label in income_stmt.index:
                            sga = income_stmt.loc[label].iloc[0]
                            break
                    rev = None
                    for label in ['Total Revenue', 'Operating Revenue']:
                        if label in income_stmt.index:
                            rev = income_stmt.loc[label].iloc[0]
                            break
                    if sga is not None and rev is not None and rev > 0:
                        metrics['sga_pct'] = (sga / rev) * 100
                    else:
                        metrics['sga_pct'] = None
                else:
                    metrics['sga_pct'] = None
            except:
                metrics['sga_pct'] = None
        except:
            metrics['gross_margin'] = None
            metrics['operating_margin'] = None
            metrics['profit_margin'] = None
            metrics['fcf_margin'] = None
            metrics['sga_pct'] = None

        # ----------------------------------------------------------
        # BALANCE SHEET STRENGTH
        # ----------------------------------------------------------
        try:
            metrics['debt_to_equity'] = info.get('debtToEquity', None)
            if metrics['debt_to_equity'] is not None:
                metrics['debt_to_equity'] = metrics['debt_to_equity'] / 100  # Yahoo gives as percentage

            # FCF / Financial Debt
            fcf = info.get('freeCashflow', None)
            total_debt_val = info.get('totalDebt', None)
            if fcf is not None and total_debt_val is not None and total_debt_val > 0:
                metrics['fcf_to_debt'] = fcf / total_debt_val
            elif fcf is not None and (total_debt_val is None or total_debt_val == 0):
                metrics['fcf_to_debt'] = 5.0  # No debt = excellent
            else:
                metrics['fcf_to_debt'] = None

            # Net Cash / Net Debt Position (Chris Hohn metric)
            total_cash = info.get('totalCash', None)
            if total_cash is not None and total_debt_val is not None:
                net_cash = total_cash - total_debt_val
                mcap = info.get('marketCap', None)
                if mcap and mcap > 0:
                    metrics['net_cash_pct_mcap'] = (net_cash / mcap) * 100
                else:
                    metrics['net_cash_pct_mcap'] = None
            else:
                metrics['net_cash_pct_mcap'] = None
        except:
            metrics['debt_to_equity'] = None
            metrics['fcf_to_debt'] = None
            metrics['net_cash_pct_mcap'] = None

        # ----------------------------------------------------------
        # GROWTH METRICS (Peter Lynch & Philip Fisher)
        # ----------------------------------------------------------
        try:
            metrics['revenue_growth'] = info.get('revenueGrowth', None)
            if metrics['revenue_growth'] is not None:
                metrics['revenue_growth'] = metrics['revenue_growth'] * 100

            metrics['earnings_growth'] = info.get('earningsGrowth', None)
            if metrics['earnings_growth'] is not None:
                metrics['earnings_growth'] = metrics['earnings_growth'] * 100

            # EPS growth from financial statements
            try:
                if not income_stmt.empty:
                    eps_vals = []
                    if 'Diluted EPS' in income_stmt.index:
                        for i in range(min(4, len(income_stmt.columns))):
                            val = income_stmt.loc['Diluted EPS'].iloc[i]
                            if val is not None and not pd.isna(val):
                                eps_vals.append(float(val))
                    elif 'Basic EPS' in income_stmt.index:
                        for i in range(min(4, len(income_stmt.columns))):
                            val = income_stmt.loc['Basic EPS'].iloc[i]
                            if val is not None and not pd.isna(val):
                                eps_vals.append(float(val))

                    if len(eps_vals) >= 2 and eps_vals[-1] != 0:
                        metrics['eps_growth_multi_yr'] = ((eps_vals[0] / abs(eps_vals[-1])) - 1) * 100
                    else:
                        metrics['eps_growth_multi_yr'] = None
                else:
                    metrics['eps_growth_multi_yr'] = None
            except:
                metrics['eps_growth_multi_yr'] = None

        except:
            metrics['revenue_growth'] = None
            metrics['earnings_growth'] = None
            metrics['eps_growth_multi_yr'] = None

        # ----------------------------------------------------------
        # VALUATION METRICS (Greenblatt / Druckenmiller / Lynch)
        # ----------------------------------------------------------
        try:
            # Earnings Yield = EBIT / Enterprise Value (Greenblatt Magic Formula)
            ev = info.get('enterpriseValue', None)
            if not income_stmt.empty and ev and ev > 0:
                ebit_val = None
                for label in ['EBIT', 'Operating Income']:
                    if label in income_stmt.index:
                        ebit_val = income_stmt.loc[label].iloc[0]
                        break
                if ebit_val is not None:
                    metrics['earnings_yield'] = (ebit_val / ev) * 100
                else:
                    metrics['earnings_yield'] = None
            else:
                metrics['earnings_yield'] = None

            # PEG Ratio (Peter Lynch)
            metrics['peg_ratio'] = info.get('pegRatio', None)

            # Owner Earnings Yield = FCF per share / Price
            fcf_ps = info.get('freeCashflow', None)
            shares = info.get('sharesOutstanding', None)
            price = info.get('currentPrice', info.get('previousClose', None))
            if fcf_ps is not None and shares is not None and shares > 0 and price is not None and price > 0:
                fcf_per_share = fcf_ps / shares
                metrics['owner_earnings_yield'] = (fcf_per_share / price) * 100
            else:
                metrics['owner_earnings_yield'] = None

            # Forward P/E (useful context)
            metrics['forward_pe'] = info.get('forwardPE', None)
            metrics['trailing_pe'] = info.get('trailingPE', None)

        except:
            metrics['earnings_yield'] = None
            metrics['peg_ratio'] = None
            metrics['owner_earnings_yield'] = None
            metrics['forward_pe'] = None
            metrics['trailing_pe'] = None

        return metrics

    except Exception as e:
        return None


def score_all_stocks(tickers_df):
    """
    Fetch data for all tickers, then score and rank them.
    """
    all_metrics = []
    total = len(tickers_df)
    failed = []

    print(f"\n{'='*60}")
    print(f"  FUNDAMENTAL SCORING ENGINE")
    print(f"  Processing {total} stocks...")
    print(f"{'='*60}\n")

    for idx, row in tickers_df.iterrows():
        ticker = row['Ticker']
        pct = ((idx + 1) / total) * 100
        print(f"  [{idx+1}/{total}] ({pct:.0f}%) Fetching {ticker}...", end='', flush=True)

        try:
            data = fetch_stock_data(ticker)
            if data:
                all_metrics.append(data)
                print(f" OK")
            else:
                failed.append(ticker)
                print(f" SKIP (no data)")
        except Exception as e:
            failed.append(ticker)
            print(f" ERROR ({str(e)[:40]})")

        # Small delay to be respectful to Yahoo Finance servers
        if (idx + 1) % 10 == 0:
            time.sleep(1)

        # Save progress every 50 stocks in case of interruption
        if (idx + 1) % 50 == 0 and all_metrics:
            try:
                temp_df = pd.DataFrame(all_metrics)
                temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fundamental_ratings_partial.csv')
                temp_df.to_csv(temp_file, index=False)
                print(f"  >> Progress saved ({len(all_metrics)} stocks so far)")
            except:
                pass

    if not all_metrics:
        print("\nERROR: No stock data could be retrieved. Check your internet connection.")
        return None

    print(f"\n  Successfully fetched: {len(all_metrics)} stocks")
    print(f"  Skipped (no data):   {len(failed)} stocks")
    if failed:
        print(f"  Failed tickers: {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")

    # Build DataFrame
    df = pd.DataFrame(all_metrics)

    # ----------------------------------------------------------
    # SCORING: Convert raw metrics to 1-10 scores
    # ----------------------------------------------------------
    print(f"\n  Scoring stocks on all metrics...")

    # --- ROIC & Capital Efficiency (20% weight) ---
    df['score_roic'] = df['roic'].apply(lambda x: percentile_score(x, df['roic'], higher_is_better=True))
    df['score_roic_5yr'] = df['roic_5yr_avg'].apply(lambda x: percentile_score(x, df['roic_5yr_avg'], higher_is_better=True))
    df['score_roic_trend'] = df['roic_trend'].apply(lambda x: percentile_score(x, df['roic_trend'], higher_is_better=True))
    df['score_roiic'] = df['roiic'].apply(lambda x: percentile_score(x, df['roiic'], higher_is_better=True))
    df['score_reinvestment'] = df['reinvestment_rate'].apply(lambda x: percentile_score(x, df['reinvestment_rate'], higher_is_better=True))

    roic_cols = ['score_roic', 'score_roic_5yr', 'score_roic_trend', 'score_roiic', 'score_reinvestment']
    # Within ROIC category: ROIC itself is heaviest
    roic_sub_weights = {'score_roic': 0.30, 'score_roic_5yr': 0.25, 'score_roic_trend': 0.20, 'score_roiic': 0.15, 'score_reinvestment': 0.10}

    def weighted_category_score(row, cols, sub_weights):
        valid_scores = {}
        for c in cols:
            val = row.get(c)
            if val is not None and not pd.isna(val):
                valid_scores[c] = val
        if not valid_scores:
            return None
        total_weight = sum(sub_weights[c] for c in valid_scores)
        if total_weight == 0:
            return None
        return sum(valid_scores[c] * sub_weights[c] / total_weight for c in valid_scores)

    df['cat_roic'] = df.apply(lambda row: weighted_category_score(row, roic_cols, roic_sub_weights), axis=1)

    # --- Profitability (20% weight) ---
    df['score_gross_margin'] = df['gross_margin'].apply(lambda x: percentile_score(x, df['gross_margin'], higher_is_better=True))
    df['score_operating_margin'] = df['operating_margin'].apply(lambda x: percentile_score(x, df['operating_margin'], higher_is_better=True))
    df['score_fcf_margin'] = df['fcf_margin'].apply(lambda x: percentile_score(x, df['fcf_margin'], higher_is_better=True))
    df['score_sga'] = df['sga_pct'].apply(lambda x: percentile_score(x, df['sga_pct'], higher_is_better=False))  # Lower SGA is better

    profit_cols = ['score_gross_margin', 'score_operating_margin', 'score_fcf_margin', 'score_sga']
    profit_sub_weights = {'score_gross_margin': 0.25, 'score_operating_margin': 0.30, 'score_fcf_margin': 0.30, 'score_sga': 0.15}
    df['cat_profitability'] = df.apply(lambda row: weighted_category_score(row, profit_cols, profit_sub_weights), axis=1)

    # --- Balance Sheet Strength (15% weight) ---
    df['score_debt_equity'] = df['debt_to_equity'].apply(lambda x: percentile_score(x, df['debt_to_equity'], higher_is_better=False))  # Lower is better
    df['score_fcf_debt'] = df['fcf_to_debt'].apply(lambda x: percentile_score(x, df['fcf_to_debt'], higher_is_better=True))
    df['score_net_cash'] = df['net_cash_pct_mcap'].apply(lambda x: percentile_score(x, df['net_cash_pct_mcap'], higher_is_better=True))

    bs_cols = ['score_debt_equity', 'score_fcf_debt', 'score_net_cash']
    bs_sub_weights = {'score_debt_equity': 0.35, 'score_fcf_debt': 0.35, 'score_net_cash': 0.30}
    df['cat_balance_sheet'] = df.apply(lambda row: weighted_category_score(row, bs_cols, bs_sub_weights), axis=1)

    # --- Growth (10% weight) ---
    df['score_rev_growth'] = df['revenue_growth'].apply(lambda x: percentile_score(x, df['revenue_growth'], higher_is_better=True))
    df['score_earnings_growth'] = df['earnings_growth'].apply(lambda x: percentile_score(x, df['earnings_growth'], higher_is_better=True))
    df['score_eps_growth'] = df['eps_growth_multi_yr'].apply(lambda x: percentile_score(x, df['eps_growth_multi_yr'], higher_is_better=True))

    growth_cols = ['score_rev_growth', 'score_earnings_growth', 'score_eps_growth']
    growth_sub_weights = {'score_rev_growth': 0.30, 'score_earnings_growth': 0.35, 'score_eps_growth': 0.35}
    df['cat_growth'] = df.apply(lambda row: weighted_category_score(row, growth_cols, growth_sub_weights), axis=1)

    # --- Valuation (35% weight) ---
    df['score_earnings_yield'] = df['earnings_yield'].apply(lambda x: percentile_score(x, df['earnings_yield'], higher_is_better=True))
    df['score_peg'] = df['peg_ratio'].apply(lambda x: percentile_score(x, df['peg_ratio'], higher_is_better=False))  # Lower PEG is better
    df['score_owner_yield'] = df['owner_earnings_yield'].apply(lambda x: percentile_score(x, df['owner_earnings_yield'], higher_is_better=True))

    val_cols = ['score_earnings_yield', 'score_peg', 'score_owner_yield']
    val_sub_weights = {'score_earnings_yield': 0.40, 'score_peg': 0.25, 'score_owner_yield': 0.35}
    df['cat_valuation'] = df.apply(lambda row: weighted_category_score(row, val_cols, val_sub_weights), axis=1)

    # ----------------------------------------------------------
    # FINAL FUNDAMENTAL RATING (0-100)
    # ----------------------------------------------------------
    def compute_final_rating(row):
        cats = {
            'roic_efficiency': row.get('cat_roic'),
            'profitability':   row.get('cat_profitability'),
            'balance_sheet':   row.get('cat_balance_sheet'),
            'growth':          row.get('cat_growth'),
            'valuation':       row.get('cat_valuation'),
        }
        valid = {k: v for k, v in cats.items() if v is not None and not pd.isna(v)}
        if not valid:
            return None
        total_weight = sum(CATEGORY_WEIGHTS[k] for k in valid)
        if total_weight == 0:
            return None
        weighted_score = sum(valid[k] * CATEGORY_WEIGHTS[k] / total_weight for k in valid)
        # Scale from 1-10 to 0-100
        return round((weighted_score - 1) / 9 * 100, 1)

    df['fundamental_rating'] = df.apply(compute_final_rating, axis=1)

    # Sort by rating (best first)
    df = df.sort_values('fundamental_rating', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # Start rank at 1
    df.index.name = 'Rank'

    return df, failed


def format_pct(val):
    """Format a value as percentage string."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A'
    return f"{val:.1f}%"


def format_score(val):
    """Format a score value."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A'
    return f"{val:.1f}"


def main():
    print("\n" + "="*60)
    print("  FUNDAMENTAL STOCK SCORING TOOL")
    print("  Inspired by Greenblatt, Fisher, Lynch, Hohn, Druckenmiller")
    print("="*60)

    # Read stock universe - works on both Linux and Windows
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    universe_file = os.path.join(script_dir, 'stock_universe.csv')
    try:
        tickers_df = pd.read_csv(universe_file)
        # Clean up ticker column name
        if 'Ticker' not in tickers_df.columns:
            # Try to find the ticker column
            for col in tickers_df.columns:
                if 'tick' in col.lower():
                    tickers_df = tickers_df.rename(columns={col: 'Ticker'})
                    break
        print(f"\n  Loaded {len(tickers_df)} tickers from {universe_file}")
    except Exception as e:
        print(f"\n  ERROR: Could not read {universe_file}: {e}")
        sys.exit(1)

    # Remove any duplicates (keep first occurrence)
    tickers_df = tickers_df.drop_duplicates(subset='Ticker').reset_index(drop=True)
    print(f"  Unique tickers: {len(tickers_df)}")

    # Run the scoring
    result = score_all_stocks(tickers_df)
    if result is None:
        print("\n  No results to display.")
        sys.exit(1)

    df, failed = result

    # ----------------------------------------------------------
    # OUTPUT: Save full results to CSV
    # ----------------------------------------------------------
    output_cols = [
        'ticker', 'name', 'sector', 'industry', 'fundamental_rating',
        'cat_roic', 'cat_profitability', 'cat_balance_sheet', 'cat_growth', 'cat_valuation',
        'roic', 'roic_5yr_avg', 'roic_trend', 'roiic', 'reinvestment_rate',
        'gross_margin', 'operating_margin', 'fcf_margin', 'sga_pct',
        'debt_to_equity', 'fcf_to_debt', 'net_cash_pct_mcap',
        'revenue_growth', 'earnings_growth', 'eps_growth_multi_yr',
        'earnings_yield', 'peg_ratio', 'owner_earnings_yield',
        'forward_pe', 'trailing_pe', 'market_cap'
    ]

    # Only include columns that exist
    output_cols = [c for c in output_cols if c in df.columns]

    output_df = df[output_cols].copy()

    # Rename columns for readability
    column_names = {
        'ticker': 'Ticker',
        'name': 'Company Name',
        'sector': 'Sector',
        'industry': 'Industry',
        'fundamental_rating': 'FUNDAMENTAL RATING (0-100)',
        'cat_roic': 'ROIC Score (1-10)',
        'cat_profitability': 'Profitability Score (1-10)',
        'cat_balance_sheet': 'Balance Sheet Score (1-10)',
        'cat_growth': 'Growth Score (1-10)',
        'cat_valuation': 'Valuation Score (1-10)',
        'roic': 'ROIC %',
        'roic_5yr_avg': '5yr Avg ROIC %',
        'roic_trend': 'ROIC Trend (+ improving)',
        'roiic': 'ROIIC %',
        'reinvestment_rate': 'Reinvestment Rate %',
        'gross_margin': 'Gross Margin %',
        'operating_margin': 'Operating Margin %',
        'fcf_margin': 'FCF Margin %',
        'sga_pct': 'SGA % of Revenue',
        'debt_to_equity': 'Debt/Equity',
        'fcf_to_debt': 'FCF/Debt',
        'net_cash_pct_mcap': 'Net Cash % of Mkt Cap',
        'revenue_growth': 'Revenue Growth %',
        'earnings_growth': 'Earnings Growth %',
        'eps_growth_multi_yr': 'EPS Growth (Multi-Year) %',
        'earnings_yield': 'Earnings Yield %',
        'peg_ratio': 'PEG Ratio',
        'owner_earnings_yield': 'Owner Earnings Yield %',
        'forward_pe': 'Forward P/E',
        'trailing_pe': 'Trailing P/E',
        'market_cap': 'Market Cap'
    }
    output_df = output_df.rename(columns=column_names)

    # Save to CSV - same folder as the script
    output_file = os.path.join(script_dir, 'fundamental_ratings.csv')
    output_df.to_csv(output_file)
    print(f"\n  Full results saved to: {output_file}")

    # ----------------------------------------------------------
    # DISPLAY: Top 50 and Bottom 20
    # ----------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  TOP 50 STOCKS BY FUNDAMENTAL RATING")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Ticker':<8}{'Company':<35}{'Rating':>8}{'ROIC':>7}{'Prof':>7}{'BS':>7}{'Grow':>7}{'Val':>7}")
    print(f"{'-'*80}")

    for rank, row in df.head(50).iterrows():
        name = str(row.get('name', ''))[:33]
        rating = format_score(row.get('fundamental_rating'))
        roic_s = format_score(row.get('cat_roic'))
        prof_s = format_score(row.get('cat_profitability'))
        bs_s = format_score(row.get('cat_balance_sheet'))
        gr_s = format_score(row.get('cat_growth'))
        val_s = format_score(row.get('cat_valuation'))
        print(f"{rank:<6}{row['ticker']:<8}{name:<35}{rating:>8}{roic_s:>7}{prof_s:>7}{bs_s:>7}{gr_s:>7}{val_s:>7}")

    print(f"\n{'='*80}")
    print(f"  BOTTOM 20 STOCKS BY FUNDAMENTAL RATING")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Ticker':<8}{'Company':<35}{'Rating':>8}{'ROIC':>7}{'Prof':>7}{'BS':>7}{'Grow':>7}{'Val':>7}")
    print(f"{'-'*80}")

    for rank, row in df.tail(20).iterrows():
        name = str(row.get('name', ''))[:33]
        rating = format_score(row.get('fundamental_rating'))
        roic_s = format_score(row.get('cat_roic'))
        prof_s = format_score(row.get('cat_profitability'))
        bs_s = format_score(row.get('cat_balance_sheet'))
        gr_s = format_score(row.get('cat_growth'))
        val_s = format_score(row.get('cat_valuation'))
        print(f"{rank:<6}{row['ticker']:<8}{name:<35}{rating:>8}{roic_s:>7}{prof_s:>7}{bs_s:>7}{gr_s:>7}{val_s:>7}")

    # Summary stats
    rated = df['fundamental_rating'].dropna()
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total stocks rated:    {len(rated)}")
    print(f"  Average rating:        {rated.mean():.1f}")
    print(f"  Median rating:         {rated.median():.1f}")
    print(f"  Highest rating:        {rated.max():.1f} ({df.iloc[0]['ticker']})")
    print(f"  Lowest rating:         {rated.min():.1f} ({df.iloc[-1]['ticker']})")
    print(f"\n  Category Weights Used:")
    print(f"    ROIC & Efficiency:   20%")
    print(f"    Profitability:       20%")
    print(f"    Balance Sheet:       15%")
    print(f"    Growth:              10%")
    print(f"    Valuation:           35%")
    print(f"\n  Full results file: {output_file}")
    print(f"  (Open this CSV in Excel or Google Sheets to sort and filter)\n")

    return df


if __name__ == '__main__':
    main()
