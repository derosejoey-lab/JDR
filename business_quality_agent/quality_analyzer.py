# Business Quality Agent — LLM Quality Analyzer
# Researches and summarises a stock's quality characteristics using Gemini.
#
# Dependencies: pip install google-genai pypdf pandas
# API key:      export GEMINI_API_KEY="..."
#
# Usage:
#   python quality_analyzer.py

import os
import sys
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[ERROR] 'google-genai' is not installed.")
    print("        Fix: pip install google-genai")
    sys.exit(1)

# Sibling-module imports (works whether run as a script or imported).
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config import CSV_PATH, BASE_DIR as INVESTMENT_COMMITTEE_FOLDER
from pdf_reader import load_all_pdfs
from read_universe import load_tickers
from stakeholder_sentiment import gather_stakeholder_sentiment

# ---------------------------------------------------------------------------
# Quality-investing principles distilled from:
#   • "Quality Investing" — Cunningham, Eide & Hargreaves (AKO Capital, 2016)
#   • "How to Pick Quality Shares" — Phil Oakley (2016)
# ---------------------------------------------------------------------------

QUALITY_PRINCIPLES = """
## QUALITY INVESTING FRAMEWORK

You are guided by two foundational texts on quality investing. Apply their
principles rigorously to every stock you analyse.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ECONOMIC MOATS  (Source: "Quality Investing", AKO Capital)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Five recognised moat sources — assess which apply and how durable each is:

  a) Network Effects      — value grows with each additional user/participant.
                            Best moats compound over time (e.g. payment rails,
                            marketplaces, social platforms).
  b) Switching Costs      — high cost (financial, operational, psychological) of
                            migrating to a rival. Stickiest in mission-critical
                            enterprise software, payroll, and banking systems.
  c) Intangible Assets    — proprietary brands, patents, regulatory licences,
                            or unique know-how that competitors cannot replicate.
                            Test: does the brand allow premium pricing?
  d) Cost Advantages      — structural, not temporary, cost leadership through
                            scale, proprietary processes, or favourable geography.
  e) Efficient Scale      — niche market served by one or few players; additional
                            entrants would destroy returns for everyone.

Key moat tests:
  • Can the company raise prices without meaningful volume loss?
  • Could a well-funded competitor replicate the position within 10 years?
  • Are returns on capital above the cost of capital and stable over cycles?

Red flags — absent moat:
  • Commoditised products or services with no differentiation.
  • Frequent price wars or margin compression from new entrants.
  • Revenue highly dependent on a single customer or contract.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MANAGEMENT QUALITY  (Source: "Quality Investing", AKO Capital)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Capital allocation hierarchy (in order of value creation):
  1. Reinvestment in the core business at high ROIC.
  2. Bolt-on acquisitions in adjacent markets at sensible prices.
  3. Debt repayment when leverage is elevated.
  4. Share buybacks only when the stock trades below intrinsic value.
  5. Special dividends as a last resort when no better use exists.

Alignment signals:
  • Meaningful insider ownership (founders or executives owning ≥ 3–5% of shares).
  • Remuneration tied to ROCE or FCF per share — NOT total revenue or adjusted EPS.
  • Long tenure with a clear, consistent strategic narrative.
  • Conservative accounting: no frequent "one-off" charges or adjusted metrics
    that always seem to flatter results.

Red flags — poor management:
  • Serial large acquisitions, especially at high goodwill multiples.
  • Frequent equity issuance diluting existing shareholders.
  • Excessive executive pay (> 0.5% of FCF as a rough guide).
  • Changing accounting policies or aggressive revenue recognition.
  • Over-promising and under-delivering on guidance.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. FINANCIAL RESILIENCE  (Source: "How to Pick Quality Shares", Phil Oakley)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core metrics — calculate each and compare against thresholds:

  ROCE (Return on Capital Employed)
    = EBIT / (Total Assets − Current Liabilities)
    Target: sustainably > 15%; must exceed WACC (typically 8–10%).
    Quality companies show stable or improving ROCE across economic cycles.

  Lease-Adjusted ROCE (Phil Oakley's key innovation)
    Operating leases are a financing choice, not an operating cost.
    Adjust by:
      • Adding the present value of future minimum lease payments to capital
        employed (pre-IFRS 16: disclosed in notes; post-IFRS 16: right-of-use
        assets already on balance sheet — verify completeness).
      • Adding back the implied interest component of lease costs to EBIT.
    A company with many leases (retailers, airlines) may show a much lower
    "true" ROCE once leases are treated as debt — a critical quality test.

  FCF Conversion
    = Free Cash Flow / Net Income
    FCF = Operating Cash Flow − Maintenance Capex
    Target: ≥ 80–100% consistently. A high-quality business turns almost all
    its reported profit into cash.
    Persistently < 70%: investigate aggressive accruals or high capex intensity.

  Leverage
    Net Debt / EBITDA: comfortable < 2×; > 3× warrants scrutiny.
    MUST include lease liabilities in net debt for a true picture.
    Interest Coverage (EBIT / Net Interest): ≥ 5× as a safety floor.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. HIDDEN DEBTS  (Source: "How to Pick Quality Shares", Phil Oakley)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The balance sheet often understates true indebtedness. Investigate:

  Pension Deficits
    • Gross IAS 19 deficit (not just the net balance sheet figure).
    • Express deficit as % of market capitalisation: > 20% is material.
    • Check discount rate assumptions — a 0.5% lower rate can dramatically
      increase liabilities. Conservative companies use lower discount rates.
    • Check funding ratio: < 80% funded is a concern; < 70% is serious.
    • A recovering deficit absorbs cash that would otherwise compound for equity
      holders — it is equivalent to debt service.

  Operating Leases (Off-Balance-Sheet Leverage)
    • Pre-IFRS 16: disclosed only in notes as future minimum lease payments.
      Capitalise at ~8× annual rental charge as a rough proxy for PV of leases.
    • Post-IFRS 16: right-of-use assets and lease liabilities appear on balance
      sheet — but verify all lease types are captured.
    • Retailers, restaurant chains, airlines, and telecoms are particularly
      exposed. Always restate ROCE and net debt/EBITDA on a lease-adjusted basis.

  Goodwill and Acquired Intangibles
    • Large goodwill (> 50% of net assets) signals acquisitions made at premium.
    • Impairment risk: if underlying business deteriorates, goodwill write-downs
      can devastate equity — but are non-cash, so watch for them in FCF analysis.
    • Amortisation of acquired intangibles can create a persistent gap between
      statutory EPS and adjusted EPS — assess which is more meaningful.

  Contingent Liabilities
    • Legal provisions, environmental liabilities, warranty obligations.
    • Deferred revenue that may require future service delivery.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. EARNINGS QUALITY  (Source: "How to Pick Quality Shares", Phil Oakley)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Profits that are not backed by cash are unreliable. Test rigorously:

  Accrual Ratio
    = (Net Income − Operating FCF) / Average Total Assets
    Should be close to zero. Persistently positive (profits > cash) signals that
    earnings are being accrued rather than collected — a classic earnings-quality
    red flag.

  Working Capital Trends
    • Receivables growing faster than revenue: customers are slow to pay —
      revenue may be recognised prematurely.
    • Inventory growing faster than COGS: goods are piling up unsold —
      may signal demand weakness or overproduction.
    • Payables rising faster than COGS: company is stretching suppliers —
      a short-term cash boost that is unsustainable.

  Capex vs. Depreciation
    • Capex persistently < Depreciation: the business is under-investing and
      consuming its own asset base.
    • Capex persistently >> Depreciation: may signal aggressive capitalisation
      of costs that should be expensed, artificially boosting profits.

  Adjusted vs. Statutory EPS Gap
    • Frequent "exceptional" or "one-off" charges that recur year after year are
      not exceptional — they are part of the cost of running the business.
    • The wider the gap between adjusted EPS and statutory EPS over a five-year
      period, the lower the quality of reported earnings.
""".strip()


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_stock(ticker: str) -> str:
    """Research and summarise quality characteristics for *ticker*.

    Calls the Gemini API with a rich system prompt incorporating the
    quality-investing framework, supplemented by any PDF text found in
    the AI_Investment_Committee folder.

    Returns the LLM response as a plain string.
    """
    # 1. Attempt to load supplementary PDF context.
    print(f"Preparing analysis for: {ticker}")
    print("-" * 50)
    pdf_text = load_all_pdfs(INVESTMENT_COMMITTEE_FOLDER)

    # 2. Build the system prompt.
    system_sections = [
        "You are a senior quality-focused investment analyst with 20+ years of "
        "experience evaluating businesses through the lens of long-term quality "
        "investing. You adhere strictly to the frameworks described below.",
        "",
        QUALITY_PRINCIPLES,
    ]

    if pdf_text:
        system_sections += [
            "",
            "━" * 76,
            "SUPPLEMENTARY READING FROM YOUR LIBRARY",
            "━" * 76,
            "The following text has been extracted from quality-investing reference "
            "books in your library. Use this as additional context and evidence when "
            "assessing stocks. Where the books provide specific examples or data "
            "points relevant to the ticker you are analysing, reference them.",
            "",
            pdf_text,
        ]

    system_prompt = "\n".join(system_sections)

    # 3. Build the user message.
    user_message = f"""Please research {ticker} and produce a structured quality assessment.

For each of the six sections below, give a concise but substantive evaluation.
Cite specific financial data (margins, ROCE, FCF conversion, leverage ratios,
pension status, etc.) where you have reliable knowledge of this company.
Clearly flag where your knowledge may be limited or where live data would be
needed for a definitive conclusion.

---

## Quality Assessment: {ticker}

**1. Economic Moat**
   - What type(s) of moat does {ticker} possess?
   - How wide and durable is the moat?
   - Key evidence for or against a durable competitive advantage.

**2. Management Quality**
   - Capital allocation track record.
   - Alignment of management incentives with shareholders.
   - Notable green flags or red flags in governance and strategy.

**3. Financial Resilience**
   - ROCE (approximate, and trend direction).
   - Lease-adjusted considerations if relevant.
   - FCF conversion quality.
   - Leverage (Net Debt / EBITDA, interest coverage).

**4. Hidden Debts**
   - Pension deficit status (materiality relative to market cap).
   - Operating lease obligations and their balance-sheet impact.
   - Goodwill / intangibles risk.
   - Any notable contingent liabilities.

**5. Earnings Quality**
   - Cash conversion / accrual ratio assessment.
   - Working capital discipline.
   - Capex vs. depreciation relationship.
   - Gap between adjusted and statutory earnings (if applicable).

**6. Overall Quality Verdict**
   - Rating: High / Medium / Low quality business.
   - 2–3 sentence rationale summarising the key drivers of your verdict.
"""

    # 4. Call the Gemini API.
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n[ERROR] GEMINI_API_KEY environment variable is not set.")
        print("        Fix: export GEMINI_API_KEY='...'")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print("Calling Gemini API...")
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1500,
        ),
    )

    return response.text


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Load the stock universe and pick the first ticker.
    if not CSV_PATH.exists():
        print(f"\n[ERROR] stock_universe.csv not found at: {CSV_PATH}")
        print("        Run read_universe.py first to confirm the file path.\n")
        sys.exit(1)

    tickers = load_tickers(CSV_PATH)

    if not tickers:
        print("\n[ERROR] No tickers found in the CSV. Cannot proceed.\n")
        sys.exit(1)

    first_ticker = tickers[0]

    # Run the analysis.
    try:
        result = analyze_stock(first_ticker)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error during analysis: {exc}\n")
        sys.exit(1)

    # Display the quality analysis result.
    banner = f"  QUALITY ANALYSIS: {first_ticker}  "
    border = "=" * (len(banner) + 4)
    print(f"\n{border}")
    print(f"= {banner} =")
    print(f"{border}\n")
    print(result)
    print(f"\n{border}\n")

    # --- Step 2: Stakeholder Sentiment (live web search) ---
    s_banner = f"  STAKEHOLDER SENTIMENT: {first_ticker}  "
    s_border = "=" * (len(s_banner) + 4)
    print(f"\n{s_border}")
    print(f"= {s_banner} =")
    print(f"{s_border}\n")

    api_key = os.environ.get("GEMINI_API_KEY")
    sentiment_client = genai.Client(api_key=api_key)

    try:
        sentiment = gather_stakeholder_sentiment(first_ticker, sentiment_client)
        print(sentiment)
    except EnvironmentError as exc:
        print(exc)
    except Exception as exc:
        print(f"[WARNING] Stakeholder sentiment step failed: {exc}")

    print(f"\n{s_border}\n")


if __name__ == "__main__":
    main()
