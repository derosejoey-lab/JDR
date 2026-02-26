# Business Quality Agent — Gemini Function-Calling Scoring Engine
# Scores a single ticker on the 100-point Business Quality framework using
# Gemini's structured function calling to guarantee schema-valid output.
#
# Dependencies: pip install google-genai
# API keys:     export GEMINI_API_KEY="..."

import sys
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[ERROR] 'google-genai' is not installed.  Fix: pip install google-genai")
    sys.exit(1)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# Stakeholder sentiment (Tavily) removed — points redistributed to other pillars.

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
1. DURABLE COMPETITIVE ADVANTAGE / MOAT — 40 Points
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

Low-Cost Operations / Scale Advantage [0–10 pts]
  10 = Structural, not cyclical, cost leadership. Scale so large that unit
       economics cannot be replicated by any plausible competitor. Operating
       leverage compounds as the business grows.
   5 = Some scale advantage but competitors can close the gap with capital.
   0 = No discernible cost advantage; cost structure in line with peers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. MANAGEMENT & GOVERNANCE — 30 Points
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Capital Allocation [0–20 pts]
  20 = Consistently exceptional ROCE (>20% sustained over 10 years). Capital
       deployed rationally: organic reinvestment first, opportunistic buybacks
       when undervalued, bolt-on M&A at sensible prices. No history of
       empire-building or value-destructive mega-deals. Shareholder returns
       compound at the business's underlying ROIC.
  13 = Good ROCE (15–20%). Sound capital allocation with minor blemishes
       (one acquisition that diluted returns temporarily, for example).
   6 = Average ROCE (10–15%) or inconsistent. Some questionable capital
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
3. FINANCIAL & STRUCTURAL RESILIENCE — 30 Points
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

Disruption & Cyclical Risk [0–10 pts]
  10 = Business is fundamentally "tech-proof" or actively benefiting from
       technology trends. Structurally growing industry with secular tailwinds.
       Product is a necessity, not a discretionary luxury.
   5 = Moderate disruption exposure; business is adapting but faces real
       technology or structural headwinds. Some cyclicality but not extreme.
   0 = High disruption risk (facing an existential technology threat, obsolete
       product category, or heavily cyclical industry riding a temporary boom).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Submit your assessment using the record_quality_score function. You MUST call
this function — do not respond with prose instead.
""".strip()


# ---------------------------------------------------------------------------
# Function declaration — enforces structured JSON output via Gemini function calling
# ---------------------------------------------------------------------------

SCORING_FUNCTION = types.FunctionDeclaration(
    name="record_quality_score",
    description=(
        "Record the complete structured quality score for a stock ticker. "
        "Call this function once with all scores and rationales populated. "
        "Use 'Data Unavailable' for extracted_metrics fields that cannot "
        "be confirmed. Do NOT invent financial figures."
    ),
    parameters_json_schema={
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
            "identified_moats", "identified_risks",
            "extracted_metrics",
            "one_sentence_rationale", "summary_paragraph",
        ],
        "properties": {
            "company_name": {
                "type": "string",
                "description": "Full legal company name (e.g. 'Apple Inc.')",
            },
            # ── MOAT (40 pts) ──────────────────────────────────────────────
            "network_effects_score": {
                "type": "number",
                "description": "Network Effects sub-score (0–10)",
            },
            "network_effects_rationale": {
                "type": "string",
                "description": "1–2 sentence justification for network_effects_score",
            },
            "pricing_power_score": {
                "type": "number",
                "description": "Pricing Power & Brand Heritage sub-score (0–10)",
            },
            "pricing_power_rationale": {"type": "string"},
            "switching_costs_score": {
                "type": "number",
                "description": "Switching Costs / Recurring Revenue sub-score (0–10)",
            },
            "switching_costs_rationale": {"type": "string"},
            "low_cost_ops_score": {
                "type": "number",
                "description": "Low-Cost Operations / Scale Advantage sub-score (0–10)",
            },
            "low_cost_ops_rationale": {"type": "string"},
            # ── MANAGEMENT (30 pts) ────────────────────────────────────────
            "capital_allocation_score": {
                "type": "number",
                "description": "Capital Allocation sub-score (0–20)",
            },
            "capital_allocation_rationale": {"type": "string"},
            "owner_operator_score": {
                "type": "number",
                "description": "Owner-Operator Alignment sub-score (0–10)",
            },
            "owner_operator_rationale": {"type": "string"},
            # ── RESILIENCE (30 pts) ────────────────────────────────────────
            "cash_generation_score": {
                "type": "number",
                "description": "Cash Generation / FCF Quality sub-score (0–10)",
            },
            "cash_generation_rationale": {"type": "string"},
            "hidden_debts_score": {
                "type": "number",
                "description": "Hidden Debts & Balance Sheet sub-score (0–10)",
            },
            "hidden_debts_rationale": {"type": "string"},
            "disruption_risk_score": {
                "type": "number",
                "description": "Disruption & Cyclical Risk sub-score (0–10)",
            },
            "disruption_risk_rationale": {"type": "string"},
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
                },
                "required": ["roce", "fcf_conversion", "net_debt_ebitda"],
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
)


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_ticker(
    ticker: str,
    client: genai.Client,
) -> dict:
    """Score *ticker* using the 100-point Business Quality framework.

    Calls Gemini with forced function calling on `record_quality_score`,
    extracting a guaranteed-schema-valid structured dict. Derives aggregate
    pillar totals in Python (not trusted to the LLM) for auditability.

    Returns a complete result dict.
    On any failure, returns an error-sentinel dict so the batch continues.
    """
    try:
        # 1. Build user message.
        user_message = (
            f"Score {ticker} on the Business Quality Framework and submit your "
            f"assessment using the record_quality_score function.\n\n"
            f"Use your training knowledge of {ticker}'s business fundamentals, "
            f"financial track record, competitive position, and management history "
            f"to score each sub-criterion. If a specific metric cannot be confirmed, "
            f"record 'Data Unavailable' in extracted_metrics and score conservatively."
        )

        # 2. Call Gemini — force the specific function.
        tool = types.Tool(function_declarations=[SCORING_FUNCTION])

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SCORING_SYSTEM_PROMPT,
                max_output_tokens=2000,
                tools=[tool],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["record_quality_score"],
                    ),
                ),
            ),
        )

        # 3. Extract the function call args.
        raw = dict(response.function_calls[0].args)

        # 4. Compute pillar totals in Python — never trust the LLM's arithmetic.
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
        total_score = moat_total + management_total + resilience_total

        # 5. Assemble the full result dict.
        return {
            "ticker": ticker,
            "error": False,
            "company_name": raw["company_name"],
            "total_score": round(total_score, 2),
            # Pillar totals
            "moat_total":         round(moat_total, 2),
            "management_total":   round(management_total, 2),
            "resilience_total":   round(resilience_total, 2),
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
            # Metadata
            "identified_moats":       list(raw.get("identified_moats", [])),
            "identified_risks":       list(raw.get("identified_risks", [])),
            "extracted_metrics":      dict(raw.get("extracted_metrics", {})),
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
            "network_effects_score": 0.0, "network_effects_rationale": "",
            "pricing_power_score": 0.0, "pricing_power_rationale": "",
            "switching_costs_score": 0.0, "switching_costs_rationale": "",
            "low_cost_ops_score": 0.0, "low_cost_ops_rationale": "",
            "capital_allocation_score": 0.0, "capital_allocation_rationale": "",
            "owner_operator_score": 0.0, "owner_operator_rationale": "",
            "cash_generation_score": 0.0, "cash_generation_rationale": "",
            "hidden_debts_score": 0.0, "hidden_debts_rationale": "",
            "disruption_risk_score": 0.0, "disruption_risk_rationale": "",
            "identified_moats": [],
            "identified_risks": [f"SCORING ERROR: {exc}"],
            "extracted_metrics": {
                "roce": "Data Unavailable",
                "fcf_conversion": "Data Unavailable",
                "net_debt_ebitda": "Data Unavailable",
            },
            "one_sentence_rationale": f"Scoring failed: {exc}",
            "summary_paragraph": "",
        }
