# Business Quality Agent — Claude Tool-Use Scoring Engine
# Scores a single ticker on the 100-point Business Quality framework using
# Claude's structured tool-use to guarantee schema-valid output.
#
# Dependencies: pip install anthropic tavily-python
# API keys:     export ANTHROPIC_API_KEY="sk-ant-..."
#               export TAVILY_API_KEY="tvly-..."

import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("[ERROR] 'anthropic' is not installed.  Fix: pip install anthropic")
    sys.exit(1)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from stakeholder_sentiment import _format_results_for_prompt

# ---------------------------------------------------------------------------
# Scoring rubric — embedded verbatim from the investment committee spec
# ---------------------------------------------------------------------------

SCORING_SYSTEM_PROMPT = """
You are the Business Quality Agent, a specialized AI operating within a
multi-agent investment committee. Your sole responsibility is to evaluate
the fundamental business quality of publicly traded equities.

You operate strictly on the principles of Warren Buffett and Benjamin Graham:
durable competitive advantage, honest management, conservative financing, and
the purchase of businesses — not stocks.

STRICT RULES:
1. ZERO HALLUCINATION: If a specific metric (ROCE, FCF figure, Glassdoor rating,
   Trustpilot score) cannot be confirmed from your training knowledge or the
   web data provided, you MUST record it as "Data Unavailable" and penalise
   the relevant sub-score conservatively (score toward the lower half of range).
2. NO NEXT-MONDAY OPTIMISM: Do not award points for promised turnarounds,
   new management plans, or analyst expectations. Score only on demonstrated,
   multi-year historical performance.
3. IGNORE PRICE AND VALUATION: P/E multiples, current share price, and
   technical chart patterns are irrelevant. Focus purely on business economics.

SCORING RUBRIC — 100 POINTS TOTAL:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. DURABLE COMPETITIVE ADVANTAGE / MOAT — 35 Points
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Network Effects [0–10 pts]
  10 = Platform with deeply entrenched, multi-sided network that grows in value
       with every new participant (e.g. payment networks, dominant marketplaces).
       Competitors cannot replicate the installed base — only displacement by a
       superior network would threaten it.
   5 = Moderate network effects; meaningful but single-sided or regional.
   0 = No network effects; product is fungible or standalone.

Pricing Power & Brand Heritage [0–10 pts]
  10 = Iconic, irreplicable brand with demonstrated ability to raise prices above
       cost inflation year after year without volume loss. Consumer trust built
       over decades. Would require a generation to rebuild from scratch.
   5 = Recognisable brand with some pricing latitude; competitors offer credible
       alternatives. Pricing power present but not dominant.
   0 = Commodity offering; competing on price only. Brand is interchangeable.

Switching Costs / Recurring Revenue [0–10 pts]
  10 = Product is a "toll road" deeply embedded in customer workflows (e.g.
       mission-critical enterprise software, payroll, core banking systems).
       Switching would require months of effort and material operational risk.
       Recurring/subscription revenue dominates.
   5 = Some integration, but switching is feasible within weeks. Mix of
       recurring and transactional revenue.
   0 = Purely transactional; customers can switch on their next purchase.

Low-Cost Operations / Scale Advantage [0–5 pts]
   5 = Structural, not cyclical, cost leadership. Scale so large that unit
       economics cannot be replicated by any plausible competitor. Operating
       leverage compounds as the business grows.
   2 = Some scale advantage but competitors can close the gap with capital.
   0 = No discernible cost advantage; cost structure in line with peers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MANAGEMENT & GOVERNANCE — 25 Points
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Capital Allocation [0–15 pts]
  15 = Consistently exceptional ROCE (>20% sustained over 10 years). Capital
       deployed rationally: organic reinvestment first, opportunistic buybacks
       when undervalued, bolt-on M&A at sensible prices. No history of
       empire-building or value-destructive mega-deals. Shareholder returns
       compound at the business's underlying ROIC.
  10 = Good ROCE (15–20%). Sound capital allocation with minor blemishes
       (one acquisition that diluted returns temporarily, for example).
   5 = Average ROCE (10–15%) or inconsistent. Some questionable capital
       deployment (equity issuances, acquisitions that stalled).
   0 = Poor ROCE (<10%) or a track record of destroying shareholder value
       through reckless debt-funded M&A, persistent dilution, or serial
       restructuring charges.

Owner-Operator Alignment [0–10 pts]
  10 = Significant insider ownership (>5% of shares held by management or
       founding family). Executive pay tied to ROCE or FCF per share — NOT
       revenue or adjusted EPS. Management communicates with refreshing candour:
       openly admits mistakes, provides honest long-term guidance.
   5 = Moderate insider ownership (1–5%) or some performance alignment but
       mixed signals on candour.
   0 = Negligible insider ownership. Pay structures reward headline EPS or
       revenue growth. Earnings calls rely on evasive jargon and "adjusted"
       metrics that perpetually flatter. Management track record of over-promising.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. FINANCIAL & STRUCTURAL RESILIENCE — 25 Points
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cash Generation / FCF Quality [0–10 pts]
  10 = FCF conversion consistently ≥90% of net income over 5+ years. Capital-
       light model or disciplined capex. Working capital is a source of cash,
       not a drain. Accrual ratio near zero (earnings are fully cash-backed).
   5 = FCF conversion 70–90%. Capex-intensive but earning adequate returns.
       Some working capital volatility but manageable.
   0 = FCF conversion persistently <70%, or FCF is frequently negative while
       reported profits are positive. Heavy capitalisation of costs, rising
       receivables, or acquisitions mask the true cash economics.

Hidden Debts & Balance Sheet [0–10 pts]
  10 = Net cash position (or very modest leverage <1× EBITDA). No material
       pension deficit. Operating lease obligations modest and well-disclosed.
       Goodwill is a small fraction of net assets; no impairment risk. Clean,
       transparent balance sheet with no significant contingent liabilities.
   5 = Moderate leverage (1–2× net debt/EBITDA). Some off-balance-sheet lease
       exposure or a modest pension obligation (<10% of market cap). Goodwill
       present but acquisition track record is sound.
   0 = High leverage (>3× net debt/EBITDA), or a pension deficit >20% of
       market cap, or massive operating lease obligations that disguise the
       true capital intensity. Goodwill write-down risk is material.

Disruption & Cyclical Risk [0–5 pts]
   5 = Business is fundamentally "tech-proof" or actively benefiting from
       technology trends. Structurally growing industry with secular tailwinds.
       Product is a necessity, not a discretionary luxury.
   2 = Moderate disruption exposure; business is adapting but faces real
       technology or structural headwinds. Some cyclicality but not extreme.
   0 = High disruption risk (facing an existential technology threat, obsolete
       product category, or heavily cyclical industry riding a temporary boom).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. STAKEHOLDER SATISFACTION — 15 Points
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Employee Sentiment [0–7.5 pts]
  Score is based ONLY on the web search data provided. Do not invent or
  assume ratings. If data is absent, score conservatively (≤3 pts).
  7.5 = Glassdoor ≥4.2/5, high CEO approval (>85%), positive culture themes,
        low attrition signals. Employees describe it as a top-tier workplace.
  4.0 = Glassdoor 3.5–4.1, mixed reviews, average CEO approval (65–85%).
        Some recurring complaints but nothing structural.
  1.0 = Glassdoor <3.5, or frequent themes of poor culture, toxic leadership,
        high attrition, or significant layoff trauma in recent reviews.
  0.0 = "Data Unavailable" — no reliable data found; score conservatively.

Customer Sentiment [0–7.5 pts]
  Score is based ONLY on the web search data provided. Do not invent or
  assume ratings. If data is absent, score conservatively (≤3 pts).
  7.5 = Trustpilot ≥4.2/5 or equivalent; consistent praise for product
        quality, reliability, and service. Brand trusted by customers.
  4.0 = Trustpilot 3.5–4.1 or mixed signals; reasonable satisfaction but
        recurring complaints about specific product lines or service failures.
  1.0 = Trustpilot <3.5 or widespread, structural customer dissatisfaction
        (product failures, poor service, brand trust eroding).
  0.0 = "Data Unavailable" — no reliable data found; score conservatively.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Submit your assessment using the record_quality_score tool. You MUST call
this tool — do not respond with prose instead.
""".strip()


# ---------------------------------------------------------------------------
# Tool definition — enforces structured JSON output via Anthropic tool use
# ---------------------------------------------------------------------------

SCORING_TOOL = {
    "name": "record_quality_score",
    "description": (
        "Record the complete structured quality score for a stock ticker. "
        "Call this tool once with all scores and rationales populated. "
        "Use 'Data Unavailable' for extracted_metrics fields that cannot "
        "be confirmed. Do NOT invent financial figures."
    ),
    "input_schema": {
        "type": "object",
        "required": [
            "company_name",
            "network_effects_score", "network_effects_rationale",
            "pricing_power_score", "pricing_power_rationale",
            "switching_costs_score", "switching_costs_rationale",
            "low_cost_ops_score", "low_cost_ops_rationale",
            "capital_allocation_score", "capital_allocation_rationale",
            "owner_operator_score", "owner_operator_rationale",
            "cash_generation_score", "cash_generation_rationale",
            "hidden_debts_score", "hidden_debts_rationale",
            "disruption_risk_score", "disruption_risk_rationale",
            "employee_sentiment_score", "employee_sentiment_rationale",
            "customer_sentiment_score", "customer_sentiment_rationale",
            "identified_moats", "identified_risks",
            "extracted_metrics",
            "one_sentence_rationale", "summary_paragraph",
        ],
        "properties": {
            "company_name": {
                "type": "string",
                "description": "Full legal company name (e.g. 'Apple Inc.')",
            },
            # ── MOAT (35 pts) ──────────────────────────────────────────────
            "network_effects_score": {
                "type": "number", "minimum": 0, "maximum": 10,
                "description": "Network Effects sub-score (0–10)",
            },
            "network_effects_rationale": {
                "type": "string",
                "description": "1–2 sentence justification for network_effects_score",
            },
            "pricing_power_score": {
                "type": "number", "minimum": 0, "maximum": 10,
                "description": "Pricing Power & Brand Heritage sub-score (0–10)",
            },
            "pricing_power_rationale": {"type": "string"},
            "switching_costs_score": {
                "type": "number", "minimum": 0, "maximum": 10,
                "description": "Switching Costs / Recurring Revenue sub-score (0–10)",
            },
            "switching_costs_rationale": {"type": "string"},
            "low_cost_ops_score": {
                "type": "number", "minimum": 0, "maximum": 5,
                "description": "Low-Cost Operations / Scale Advantage sub-score (0–5)",
            },
            "low_cost_ops_rationale": {"type": "string"},
            # ── MANAGEMENT (25 pts) ────────────────────────────────────────
            "capital_allocation_score": {
                "type": "number", "minimum": 0, "maximum": 15,
                "description": "Capital Allocation sub-score (0–15)",
            },
            "capital_allocation_rationale": {"type": "string"},
            "owner_operator_score": {
                "type": "number", "minimum": 0, "maximum": 10,
                "description": "Owner-Operator Alignment sub-score (0–10)",
            },
            "owner_operator_rationale": {"type": "string"},
            # ── RESILIENCE (25 pts) ────────────────────────────────────────
            "cash_generation_score": {
                "type": "number", "minimum": 0, "maximum": 10,
                "description": "Cash Generation / FCF Quality sub-score (0–10)",
            },
            "cash_generation_rationale": {"type": "string"},
            "hidden_debts_score": {
                "type": "number", "minimum": 0, "maximum": 10,
                "description": "Hidden Debts & Balance Sheet sub-score (0–10)",
            },
            "hidden_debts_rationale": {"type": "string"},
            "disruption_risk_score": {
                "type": "number", "minimum": 0, "maximum": 5,
                "description": "Disruption & Cyclical Risk sub-score (0–5)",
            },
            "disruption_risk_rationale": {"type": "string"},
            # ── SATISFACTION (15 pts) ──────────────────────────────────────
            "employee_sentiment_score": {
                "type": "number", "minimum": 0, "maximum": 7.5,
                "description": "Employee Sentiment sub-score (0–7.5). Must be based on provided web data only.",
            },
            "employee_sentiment_rationale": {"type": "string"},
            "customer_sentiment_score": {
                "type": "number", "minimum": 0, "maximum": 7.5,
                "description": "Customer Sentiment sub-score (0–7.5). Must be based on provided web data only.",
            },
            "customer_sentiment_rationale": {"type": "string"},
            # ── METADATA ──────────────────────────────────────────────────
            "identified_moats": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific moat sources identified (e.g. 'iOS ecosystem switching costs')",
            },
            "identified_risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific risks identified (e.g. 'Pension deficit 18% of market cap')",
            },
            "extracted_metrics": {
                "type": "object",
                "description": "Key financial metrics. Use 'Data Unavailable' for anything unconfirmed.",
                "properties": {
                    "roce":              {"type": "string"},
                    "fcf_conversion":    {"type": "string"},
                    "net_debt_ebitda":   {"type": "string"},
                    "glassdoor_rating":  {"type": "string"},
                    "trustpilot_rating": {"type": "string"},
                },
                "required": ["roce", "fcf_conversion", "net_debt_ebitda",
                             "glassdoor_rating", "trustpilot_rating"],
            },
            "one_sentence_rationale": {
                "type": "string",
                "description": "Single sentence (≤25 words) capturing the core quality thesis or key risk.",
            },
            "summary_paragraph": {
                "type": "string",
                "description": "3–5 sentence balanced summary of quality strengths and risks.",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_ticker(
    ticker: str,
    employee_results: list[dict],
    customer_results: list[dict],
    client: anthropic.Anthropic,
) -> dict:
    """Score *ticker* using the 100-point Business Quality framework.

    Calls Claude with tool_choice forced to `record_quality_score`, extracting
    a guaranteed-schema-valid structured dict. Derives aggregate pillar totals
    in Python (not trusted to Claude) for auditability.

    Returns a complete result dict.
    On any failure, returns an error-sentinel dict so the batch continues.
    """
    try:
        # 1. Format sentiment data for the prompt.
        employee_block = _format_results_for_prompt(employee_results)
        customer_block = _format_results_for_prompt(customer_results)

        # 2. Build user message.
        user_message = (
            f"Score {ticker} on the Business Quality Framework and submit your "
            f"assessment using the record_quality_score tool.\n\n"
            f"CRITICAL — Satisfaction sub-scores:\n"
            f"  Base employee_sentiment_score ONLY on the EMPLOYEE REVIEW DATA below.\n"
            f"  Base customer_sentiment_score ONLY on the CUSTOMER REVIEW DATA below.\n"
            f"  If a rating figure cannot be found in the data, record 'Data Unavailable'\n"
            f"  in extracted_metrics and score conservatively (do not exceed 3.0/7.5).\n\n"
            f"{'=' * 60}\n"
            f"EMPLOYEE REVIEW DATA\n"
            f"{'=' * 60}\n"
            f"{employee_block}\n\n"
            f"{'=' * 60}\n"
            f"CUSTOMER REVIEW DATA\n"
            f"{'=' * 60}\n"
            f"{customer_block}"
        )

        # 3. Call Claude — force the specific tool.
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            system=SCORING_SYSTEM_PROMPT,
            tools=[SCORING_TOOL],
            tool_choice={"type": "tool", "name": "record_quality_score"},
            messages=[{"role": "user", "content": user_message}],
        )

        # 4. Extract the tool input (always present due to forced tool_choice).
        raw = next(
            b.input for b in response.content if b.type == "tool_use"
        )

        # 5. Compute pillar totals in Python — never trust Claude's arithmetic.
        moat_total = (
            float(raw["network_effects_score"])
            + float(raw["pricing_power_score"])
            + float(raw["switching_costs_score"])
            + float(raw["low_cost_ops_score"])
        )
        management_total = (
            float(raw["capital_allocation_score"])
            + float(raw["owner_operator_score"])
        )
        resilience_total = (
            float(raw["cash_generation_score"])
            + float(raw["hidden_debts_score"])
            + float(raw["disruption_risk_score"])
        )
        satisfaction_total = (
            float(raw["employee_sentiment_score"])
            + float(raw["customer_sentiment_score"])
        )
        total_score = moat_total + management_total + resilience_total + satisfaction_total

        # 6. Assemble the full result dict.
        return {
            "ticker": ticker,
            "error": False,
            "company_name": raw["company_name"],
            "total_score": round(total_score, 2),
            # Pillar totals
            "moat_total":         round(moat_total, 2),
            "management_total":   round(management_total, 2),
            "resilience_total":   round(resilience_total, 2),
            "satisfaction_total": round(satisfaction_total, 2),
            # Sub-scores
            "network_effects_score":     float(raw["network_effects_score"]),
            "network_effects_rationale": raw["network_effects_rationale"],
            "pricing_power_score":       float(raw["pricing_power_score"]),
            "pricing_power_rationale":   raw["pricing_power_rationale"],
            "switching_costs_score":     float(raw["switching_costs_score"]),
            "switching_costs_rationale": raw["switching_costs_rationale"],
            "low_cost_ops_score":        float(raw["low_cost_ops_score"]),
            "low_cost_ops_rationale":    raw["low_cost_ops_rationale"],
            "capital_allocation_score":     float(raw["capital_allocation_score"]),
            "capital_allocation_rationale": raw["capital_allocation_rationale"],
            "owner_operator_score":         float(raw["owner_operator_score"]),
            "owner_operator_rationale":     raw["owner_operator_rationale"],
            "cash_generation_score":     float(raw["cash_generation_score"]),
            "cash_generation_rationale": raw["cash_generation_rationale"],
            "hidden_debts_score":        float(raw["hidden_debts_score"]),
            "hidden_debts_rationale":    raw["hidden_debts_rationale"],
            "disruption_risk_score":     float(raw["disruption_risk_score"]),
            "disruption_risk_rationale": raw["disruption_risk_rationale"],
            "employee_sentiment_score":     float(raw["employee_sentiment_score"]),
            "employee_sentiment_rationale": raw["employee_sentiment_rationale"],
            "customer_sentiment_score":     float(raw["customer_sentiment_score"]),
            "customer_sentiment_rationale": raw["customer_sentiment_rationale"],
            # Metadata
            "identified_moats":       raw.get("identified_moats", []),
            "identified_risks":       raw.get("identified_risks", []),
            "extracted_metrics":      raw.get("extracted_metrics", {}),
            "one_sentence_rationale": raw.get("one_sentence_rationale", ""),
            "summary_paragraph":      raw.get("summary_paragraph", ""),
        }

    except Exception as exc:
        print(f"  [ERROR] Failed to score {ticker}: {exc}")
        return {
            "ticker": ticker,
            "error": True,
            "company_name": "ERROR",
            "total_score": 0.0,
            "moat_total": 0.0,
            "management_total": 0.0,
            "resilience_total": 0.0,
            "satisfaction_total": 0.0,
            "network_effects_score": 0.0, "network_effects_rationale": "",
            "pricing_power_score": 0.0, "pricing_power_rationale": "",
            "switching_costs_score": 0.0, "switching_costs_rationale": "",
            "low_cost_ops_score": 0.0, "low_cost_ops_rationale": "",
            "capital_allocation_score": 0.0, "capital_allocation_rationale": "",
            "owner_operator_score": 0.0, "owner_operator_rationale": "",
            "cash_generation_score": 0.0, "cash_generation_rationale": "",
            "hidden_debts_score": 0.0, "hidden_debts_rationale": "",
            "disruption_risk_score": 0.0, "disruption_risk_rationale": "",
            "employee_sentiment_score": 0.0, "employee_sentiment_rationale": "",
            "customer_sentiment_score": 0.0, "customer_sentiment_rationale": "",
            "identified_moats": [],
            "identified_risks": [f"SCORING ERROR: {exc}"],
            "extracted_metrics": {
                "roce": "Data Unavailable",
                "fcf_conversion": "Data Unavailable",
                "net_debt_ebitda": "Data Unavailable",
                "glassdoor_rating": "Data Unavailable",
                "trustpilot_rating": "Data Unavailable",
            },
            "one_sentence_rationale": f"Scoring failed: {exc}",
            "summary_paragraph": "",
        }
