# Business Quality Agent — Full Universe Scoring Orchestrator
# Processes all tickers in stock_universe.csv in batches of 5.
# Outputs:
#   ~/Desktop/AI_Investment_Committee/quality_rankings.csv
#   ~/Desktop/AI_Investment_Committee/agent_handoff.json
# Override base path with: export BQA_BASE_DIR="/your/path"
#
# Dependencies: pip install anthropic tavily-python pandas
# API keys:
#   export ANTHROPIC_API_KEY="sk-ant-..."
#   export TAVILY_API_KEY="tvly-..."
#
# Usage:
#   python business_quality_agent/score_universe.py          # batch mode (all tickers)
#   python business_quality_agent/score_universe.py AAPL     # single-ticker mode

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("[ERROR] 'anthropic' is not installed.  Fix: pip install anthropic")
    sys.exit(1)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import CSV_PATH, OUTPUT_FOLDER, CSV_OUTPUT, JSON_OUTPUT
from read_universe import load_tickers
from stakeholder_sentiment import fetch_employee_sentiment, fetch_customer_sentiment
from scorer import score_ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 5

AGENT_VERSION = "Business Quality Agent v1.0"
MODEL         = "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_rankings_csv(results: list[dict], path: Path) -> None:
    """Write quality_rankings.csv sorted by Total Quality Score descending.

    Columns: Ticker, Company Name, Total Quality Score, Moat Score (0-35),
             Management Score (0-25), Resilience Score (0-25),
             Satisfaction Score (0-15), 1-Sentence Rationale
    """
    fieldnames = [
        "Ticker",
        "Company Name",
        "Total Quality Score",
        "Moat Score (0-35)",
        "Management Score (0-25)",
        "Resilience Score (0-25)",
        "Satisfaction Score (0-15)",
        "1-Sentence Rationale",
    ]

    sorted_results = sorted(results, key=lambda r: r["total_score"], reverse=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted_results:
            rationale = (
                "ERROR — see console output"
                if r.get("error")
                else r.get("one_sentence_rationale", "")
            )
            writer.writerow({
                "Ticker":                  r["ticker"],
                "Company Name":            r["company_name"],
                "Total Quality Score":     f"{r['total_score']:.1f}",
                "Moat Score (0-35)":       f"{r['moat_total']:.1f}",
                "Management Score (0-25)": f"{r['management_total']:.1f}",
                "Resilience Score (0-25)": f"{r['resilience_total']:.1f}",
                "Satisfaction Score (0-15)": f"{r['satisfaction_total']:.1f}",
                "1-Sentence Rationale":    rationale,
            })


def write_agent_handoff_json(results: list[dict], path: Path) -> None:
    """Write agent_handoff.json — structured payload for the Portfolio Manager agent.

    Schema:
      metadata: { generated_at, agent, model, ticker_count, scoring_methodology }
      universe: [ { ticker, company_name, total_score, sub_scores{...},
                    identified_moats[], identified_risks[],
                    extracted_metrics{...}, one_sentence_rationale,
                    summary_paragraph } ]
    Sorted by total_score descending.
    """
    sorted_results = sorted(results, key=lambda r: r["total_score"], reverse=True)

    universe = []
    for r in sorted_results:
        entry = {
            "ticker":       r["ticker"],
            "company_name": r["company_name"],
            "total_score":  r["total_score"],
            "sub_scores": {
                "moat": {
                    "total":           r["moat_total"],
                    "network_effects": r["network_effects_score"],
                    "pricing_power":   r["pricing_power_score"],
                    "switching_costs": r["switching_costs_score"],
                    "low_cost_ops":    r["low_cost_ops_score"],
                },
                "management": {
                    "total":              r["management_total"],
                    "capital_allocation": r["capital_allocation_score"],
                    "owner_operator":     r["owner_operator_score"],
                },
                "resilience": {
                    "total":            r["resilience_total"],
                    "cash_generation":  r["cash_generation_score"],
                    "hidden_debts":     r["hidden_debts_score"],
                    "disruption_risk":  r["disruption_risk_score"],
                },
                "satisfaction": {
                    "total":              r["satisfaction_total"],
                    "employee_sentiment": r["employee_sentiment_score"],
                    "customer_sentiment": r["customer_sentiment_score"],
                },
            },
            "identified_moats":       r.get("identified_moats", []),
            "identified_risks":       r.get("identified_risks", []),
            "extracted_metrics":      r.get("extracted_metrics", {}),
            "one_sentence_rationale": r.get("one_sentence_rationale", ""),
            "summary_paragraph":      r.get("summary_paragraph", ""),
        }
        universe.append(entry)

    payload = {
        "metadata": {
            "generated_at":        datetime.now(timezone.utc).isoformat(),
            "agent":               AGENT_VERSION,
            "model":               MODEL,
            "ticker_count":        len(results),
            "scoring_methodology": (
                "4-pillar framework: "
                "Moat (35pts: Network Effects 10 + Pricing Power 10 + "
                "Switching Costs 10 + Low-Cost Ops 5) | "
                "Management (25pts: Capital Allocation 15 + Owner-Operator 10) | "
                "Resilience (25pts: Cash Generation 10 + Hidden Debts 10 + "
                "Disruption Risk 5) | "
                "Satisfaction (15pts: Employee 7.5 + Customer 7.5)"
            ),
        },
        "universe": universe,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_summary_table(results: list[dict]) -> None:
    """Print a ranked summary table to stdout."""
    sorted_r = sorted(results, key=lambda r: r["total_score"], reverse=True)

    header = (
        f"{'Rank':>4}  {'Ticker':<8}  {'Company':<30}  "
        f"{'Score':>5}  {'Moat':>5}  {'Mgmt':>5}  {'Resil':>5}  {'Satisf':>6}"
    )
    divider = "─" * len(header)

    print(f"\n{divider}")
    print(header)
    print(divider)

    for i, r in enumerate(sorted_r, start=1):
        company = r["company_name"][:29] if len(r["company_name"]) > 29 else r["company_name"]
        flag = " ⚠" if r.get("error") else ""
        print(
            f"{i:>4}.  {r['ticker']:<8}  {company:<30}  "
            f"{r['total_score']:>5.1f}  "
            f"{r['moat_total']:>5.1f}  "
            f"{r['management_total']:>5.1f}  "
            f"{r['resilience_total']:>5.1f}  "
            f"{r['satisfaction_total']:>6.1f}"
            f"{flag}"
        )

    print(divider)
    scores = [r["total_score"] for r in results if not r.get("error")]
    if scores:
        print(
            f"\n  Universe average: {sum(scores)/len(scores):.1f}/100  |  "
            f"High (≥70): {sum(1 for s in scores if s >= 70)}  |  "
            f"Medium (50–69): {sum(1 for s in scores if 50 <= s < 70)}  |  "
            f"Low (<50): {sum(1 for s in scores if s < 50)}"
        )
    errors = sum(1 for r in results if r.get("error"))
    if errors:
        print(f"  ⚠  {errors} ticker(s) failed scoring — check console output above.")
    print()


# ---------------------------------------------------------------------------
# Single-ticker mode
# ---------------------------------------------------------------------------

def _print_single_result(r: dict) -> None:
    """Print a detailed scorecard for a single ticker."""
    border = "═" * 70

    if r.get("error"):
        print(f"\n{border}")
        print(f"  {r['ticker']} — SCORING FAILED")
        print(f"{border}")
        print(f"  Error: {r.get('error_message', 'unknown error')}\n")
        return

    print(f"\n{border}")
    print(f"  {r['ticker']}  —  {r['company_name']}")
    print(f"  TOTAL QUALITY SCORE:  {r['total_score']:.1f} / 100")
    print(f"{border}")

    # Pillar totals
    print(f"\n  {'Pillar':<20} {'Score':>6}  {'Max':>4}")
    print(f"  {'─' * 34}")
    print(f"  {'Moat':<20} {r['moat_total']:>6.1f}  {'/35':>4}")
    print(f"  {'Management':<20} {r['management_total']:>6.1f}  {'/25':>4}")
    print(f"  {'Resilience':<20} {r['resilience_total']:>6.1f}  {'/25':>4}")
    print(f"  {'Satisfaction':<20} {r['satisfaction_total']:>6.1f}  {'/15':>4}")

    # Sub-scores with rationales
    sub_scores = [
        ("Moat", [
            ("Network Effects",  "network_effects",  10),
            ("Pricing Power",    "pricing_power",     10),
            ("Switching Costs",  "switching_costs",   10),
            ("Low-Cost Ops",     "low_cost_ops",       5),
        ]),
        ("Management", [
            ("Capital Allocation", "capital_allocation", 15),
            ("Owner-Operator",     "owner_operator",     10),
        ]),
        ("Resilience", [
            ("Cash Generation",  "cash_generation",  10),
            ("Hidden Debts",     "hidden_debts",     10),
            ("Disruption Risk",  "disruption_risk",   5),
        ]),
        ("Satisfaction", [
            ("Employee Sentiment", "employee_sentiment", 7.5),
            ("Customer Sentiment", "customer_sentiment", 7.5),
        ]),
    ]

    for pillar_name, items in sub_scores:
        print(f"\n  ── {pillar_name} ──")
        for label, key, max_pts in items:
            score = r.get(f"{key}_score", 0)
            rationale = r.get(f"{key}_rationale", "")
            print(f"    {label:<22} {score:>5.1f}/{max_pts}")
            if rationale:
                print(f"      {rationale}")

    # Moats and risks
    moats = r.get("identified_moats", [])
    risks = r.get("identified_risks", [])
    if moats:
        print(f"\n  Identified Moats:")
        for m in moats:
            print(f"    + {m}")
    if risks:
        print(f"\n  Identified Risks:")
        for risk in risks:
            print(f"    - {risk}")

    # Extracted metrics
    metrics = r.get("extracted_metrics", {})
    if metrics and any(v for v in metrics.values()):
        print(f"\n  Extracted Metrics:")
        for k, v in metrics.items():
            if v:
                label = k.replace("_", " ").title()
                print(f"    {label:<20} {v}")

    # Summary
    summary = r.get("summary_paragraph", "")
    if summary:
        print(f"\n  Summary:")
        print(f"    {summary}")

    print(f"\n{border}\n")


def run_single(ticker: str, client) -> None:
    """Score a single ticker and write output files."""
    print(f"\n  Fetching stakeholder sentiment for {ticker}...")
    try:
        employee_results = fetch_employee_sentiment(ticker)
        customer_results = fetch_customer_sentiment(ticker)
        print(
            f"  {len(employee_results)} employee, "
            f"{len(customer_results)} customer source(s) found."
        )
    except EnvironmentError as exc:
        print(f"\n{exc}\n")
        sys.exit(1)
    except Exception as exc:
        print(f"  Sentiment search failed: {exc}")
        employee_results, customer_results = [], []

    print(f"  Scoring {ticker} with Claude...")
    result = score_ticker(ticker, employee_results, customer_results, client)

    _print_single_result(result)

    # Write output files (single-entry lists).
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    write_rankings_csv([result], CSV_OUTPUT)
    write_agent_handoff_json([result], JSON_OUTPUT)

    print(f"  Output files:")
    print(f"    {CSV_OUTPUT}")
    print(f"    {JSON_OUTPUT}\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    # 0. Parse CLI arguments.
    parser = argparse.ArgumentParser(
        description="Business Quality Agent — score stocks on a 100-point framework.",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default=None,
        help="Score a single ticker (e.g. AAPL). Omit to score all tickers from CSV.",
    )
    args = parser.parse_args()

    # 1. Validate environment.
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        print("\n[ERROR] ANTHROPIC_API_KEY is not set.")
        print("        Fix: export ANTHROPIC_API_KEY='sk-ant-...'\n")
        sys.exit(1)

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("\n[ERROR] TAVILY_API_KEY is not set.")
        print("        Fix: export TAVILY_API_KEY='tvly-...'")
        print("        Get a free key at: https://tavily.com\n")
        sys.exit(1)

    # 1b. Single-ticker mode — skip CSV, score one stock, and exit.
    if args.ticker:
        client = anthropic.Anthropic(api_key=anthropic_key)
        border = "═" * 70
        print(f"\n{border}")
        print(f"  BUSINESS QUALITY AGENT — SINGLE TICKER")
        print(f"  {AGENT_VERSION}  |  Model: {MODEL}")
        print(f"  Ticker: {args.ticker.upper()}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{border}")
        run_single(args.ticker.upper(), client)
        return

    # 2. Load stock universe (batch mode).
    if not CSV_PATH.exists():
        print(f"\n[ERROR] Stock universe not found: {CSV_PATH}")
        print( "        Set BQA_BASE_DIR to point to your folder:")
        print( '        export BQA_BASE_DIR="/path/to/your/folder"\n')
        sys.exit(1)

    tickers = load_tickers(CSV_PATH)
    if not tickers:
        print("\n[ERROR] No tickers found in stock_universe.csv\n")
        sys.exit(1)

    # 3. Ensure output folder exists.
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # 4. Initialise the Anthropic client once for the full run.
    client = anthropic.Anthropic(api_key=anthropic_key)

    # 5. Print run header.
    n_tickers = len(tickers)
    n_batches = (n_tickers + BATCH_SIZE - 1) // BATCH_SIZE
    border = "═" * 70
    print(f"\n{border}")
    print(f"  BUSINESS QUALITY AGENT — FULL UNIVERSE SCORING")
    print(f"  {AGENT_VERSION}  |  Model: {MODEL}")
    print(f"  Tickers: {n_tickers}  |  Batch size: {BATCH_SIZE}  |  Batches: {n_batches}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{border}\n")

    # 6. Batch processing loop.
    all_results: list[dict] = []

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch       = tickers[batch_start : batch_start + BATCH_SIZE]
        batch_label = ", ".join(batch)

        print(f"━━━ BATCH {batch_idx + 1}/{n_batches} — [{batch_label}] ━━━\n")

        for ticker in batch:
            global_i = batch_start + batch.index(ticker) + 1

            # a. Stakeholder sentiment (Tavily web search).
            print(f"  [{global_i}/{n_tickers}] {ticker} — fetching sentiment data...")
            try:
                employee_results = fetch_employee_sentiment(ticker)
                customer_results = fetch_customer_sentiment(ticker)
                print(
                    f"  [{global_i}/{n_tickers}] {ticker} — "
                    f"{len(employee_results)} employee, "
                    f"{len(customer_results)} customer source(s) found."
                )
            except EnvironmentError as exc:
                # Tavily key missing — fail the entire run cleanly.
                print(f"\n{exc}\n")
                sys.exit(1)
            except Exception as exc:
                print(f"  [{global_i}/{n_tickers}] {ticker} — sentiment search failed: {exc}")
                employee_results, customer_results = [], []

            # b. Claude scoring (tool use).
            print(f"  [{global_i}/{n_tickers}] {ticker} — scoring with Claude...")
            result = score_ticker(ticker, employee_results, customer_results, client)
            all_results.append(result)

            status = f"Score: {result['total_score']:.1f}/100"
            flag   = " ⚠ ERROR" if result.get("error") else " ✓"
            print(f"  [{global_i}/{n_tickers}] {ticker}{flag}  {status}\n")

        print()  # blank line between batches

    # 7. Write outputs.
    print("Writing output files...")
    write_rankings_csv(all_results, CSV_OUTPUT)
    print(f"  ✓  quality_rankings.csv  ({len(all_results)} rows)")

    write_agent_handoff_json(all_results, JSON_OUTPUT)
    print(f"  ✓  agent_handoff.json    ({len(all_results)} entries)\n")

    # 8. Print ranked summary table.
    _print_summary_table(all_results)

    # 9. Print output paths for easy access.
    print(f"{border}")
    print(f"  OUTPUT FILES")
    print(f"{border}")
    print(f"  Excel/CSV:  {CSV_OUTPUT}")
    print(f"  JSON:       {JSON_OUTPUT}")
    print(f"{border}\n")


if __name__ == "__main__":
    main()
