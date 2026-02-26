# Business Quality Agent — Stakeholder Sentiment via Web Search
# Gathers live employee and customer sentiment data using Tavily web search,
# then synthesises it into a structured report via Gemini.
#
# Dependencies: pip install tavily-python google-genai
# API keys:
#   export TAVILY_API_KEY="tvly-..."
#   export GEMINI_API_KEY="..."

import os
import sys
from pathlib import Path

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # Handled gracefully in _get_tavily_client()

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("[ERROR] 'google-genai' is not installed.")
    print("        Fix: pip install google-genai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_RESULTS_PER_QUERY = 5   # Tavily results per search call
MAX_CONTENT_PER_RESULT = 800  # Characters of page content to include per result
MAX_SECTION_CHARS = 6_000   # Total chars for each sentiment section in the prompt
SENTIMENT_MAX_TOKENS = 800  # Gemini response cap for synthesis (lightweight call)


# ---------------------------------------------------------------------------
# Tavily client
# ---------------------------------------------------------------------------

def _get_tavily_client() -> "TavilyClient":
    """Return an initialised Tavily client.

    Raises EnvironmentError if the API key is missing or the library is absent.
    """
    if TavilyClient is None:
        raise EnvironmentError(
            "[ERROR] 'tavily-python' is not installed.\n"
            "        Fix: pip install tavily-python"
        )

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "[ERROR] TAVILY_API_KEY environment variable is not set.\n"
            "        Fix: export TAVILY_API_KEY='tvly-...'\n"
            "        Get a free key at: https://tavily.com"
        )

    return TavilyClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Core search function
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """Execute a Tavily web search and return a list of result dicts.

    Each result contains: {title, url, content, score}
    The `content` field holds extracted body text from the target page —
    much richer than a plain search snippet.

    Returns [] and prints a warning on any failure (never raises).
    """
    try:
        client = _get_tavily_client()
        response = client.search(query, max_results=max_results)
        return response.get("results", [])
    except EnvironmentError:
        raise  # Propagate key/install errors — caller handles them
    except Exception as exc:
        print(f"  [WARNING] Search failed for '{query}': {exc}")
        return []


# ---------------------------------------------------------------------------
# Targeted sentiment searches
# ---------------------------------------------------------------------------

def fetch_employee_sentiment(ticker: str) -> list[dict]:
    """Search for employee reviews on Glassdoor and similar platforms.

    Runs two queries and deduplicates by URL for maximum coverage.
    Returns a combined list of up to 2 × MAX_RESULTS_PER_QUERY results.
    """
    queries = [
        f"{ticker} Glassdoor employee reviews rating culture 2025",
        f"{ticker} employee satisfaction work environment reviews",
    ]

    seen_urls: set[str] = set()
    combined: list[dict] = []

    for query in queries:
        results = search_web(query)
        for result in results:
            url = result.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)

    return combined


def fetch_customer_sentiment(ticker: str) -> list[dict]:
    """Search for customer reviews on Trustpilot and general review sites.

    Runs two queries and deduplicates by URL for maximum coverage.
    Returns a combined list of up to 2 × MAX_RESULTS_PER_QUERY results.
    """
    queries = [
        f"{ticker} Trustpilot customer reviews rating 2025",
        f"{ticker} customer reviews complaints satisfaction",
    ]

    seen_urls: set[str] = set()
    combined: list[dict] = []

    for query in queries:
        results = search_web(query)
        for result in results:
            url = result.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)

    return combined


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_results_for_prompt(
    results: list[dict],
    max_chars: int = MAX_SECTION_CHARS,
) -> str:
    """Format a list of Tavily results into a readable block for a Gemini prompt.

    Truncates total output to *max_chars* to keep token usage bounded.
    """
    if not results:
        return "No results found."

    sections: list[str] = []
    total = 0

    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = (r.get("content") or "").strip()

        # Trim per-result content to keep the prompt manageable.
        if len(content) > MAX_CONTENT_PER_RESULT:
            content = content[:MAX_CONTENT_PER_RESULT] + "..."

        block = f"SOURCE: {title}\nURL:    {url}\n\n{content}\n---"
        remaining = max_chars - total

        if len(block) >= remaining:
            sections.append(block[:remaining])
            sections.append("\n[...additional results truncated to stay within token budget]")
            break

        sections.append(block)
        total += len(block)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Gemini synthesis
# ---------------------------------------------------------------------------

def synthesize_sentiment(
    ticker: str,
    employee_results: list[dict],
    customer_results: list[dict],
    client: genai.Client,
) -> str:
    """Use Gemini to synthesise raw search results into a structured sentiment report.

    This is a lightweight synthesis call (max_output_tokens=800).
    Returns the response text.
    """
    employee_block = _format_results_for_prompt(employee_results)
    customer_block = _format_results_for_prompt(customer_results)

    user_message = f"""You have been given raw web search results about {ticker}'s stakeholder reviews.
Synthesise them into a concise, structured Stakeholder Sentiment Report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMPLOYEE SENTIMENT  (Glassdoor / employee review platforms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report on:
  • Overall Glassdoor rating and/or trend direction (if data is present)
  • Top 3 recurring employee positives (e.g. culture, pay, career growth)
  • Top 3 recurring employee negatives / complaints
  • Any notable recent changes (layoffs, leadership changes, cultural shifts)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUSTOMER SENTIMENT  (Trustpilot / consumer review platforms)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Report on:
  • Overall Trustpilot or review-site rating and/or trend direction
  • Top 3 recurring customer positives (e.g. product quality, service, value)
  • Top 3 recurring customer negatives / complaints
  • Any notable patterns (improving, declining, specific product/service issues)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INVESTOR IMPLICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • What do these stakeholder signals suggest about moat durability,
    management quality, and long-term business health?
  • Flag any material risks or green flags surfaced by the review data.

Be concise and factual. Quote specific ratings or data points from the search
results wherever they appear. If data is missing or inconclusive, say so clearly.

════════════════════════════════════════════════════════════════════════════════
EMPLOYEE REVIEW DATA
════════════════════════════════════════════════════════════════════════════════
{employee_block}

════════════════════════════════════════════════════════════════════════════════
CUSTOMER REVIEW DATA
════════════════════════════════════════════════════════════════════════════════
{customer_block}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
            max_output_tokens=SENTIMENT_MAX_TOKENS,
        ),
    )

    return response.text


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def gather_stakeholder_sentiment(
    ticker: str,
    gemini_client: genai.Client,
) -> str:
    """Fetch and synthesise stakeholder sentiment for *ticker*.

    Orchestrates the full pipeline:
      1. Web search for employee sentiment (Glassdoor-focused)
      2. Web search for customer sentiment (Trustpilot-focused)
      3. Gemini synthesis of raw results into a structured report

    Returns the synthesised report as a plain string.
    Raises EnvironmentError if TAVILY_API_KEY is missing/invalid.
    """
    # 1. Employee sentiment.
    print("  Searching employee sentiment (Glassdoor)...")
    employee_results = fetch_employee_sentiment(ticker)
    print(f"  Found {len(employee_results)} employee review source(s).")

    # 2. Customer sentiment.
    print("  Searching customer sentiment (Trustpilot / reviews)...")
    customer_results = fetch_customer_sentiment(ticker)
    print(f"  Found {len(customer_results)} customer review source(s).")

    # 3. Guard: no results at all.
    if not employee_results and not customer_results:
        return (
            "[WARNING] No stakeholder review data was returned by the web search.\n"
            "          This may be a temporary rate-limit or network issue.\n"
            "          Try again, or check your TAVILY_API_KEY."
        )

    # 4. Gemini synthesis.
    print("  Synthesising stakeholder sentiment with Gemini...")
    return synthesize_sentiment(ticker, employee_results, customer_results, gemini_client)
