# Business Quality Agent — PDF Reader
# Extracts text from quality-investing PDFs in the AI_Investment_Committee folder.
#
# Dependency: pypdf  (install with: pip install pypdf)

from pathlib import Path

try:
    import pypdf
except ImportError:
    pypdf = None  # Handled gracefully in load_all_pdfs()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INVESTMENT_COMMITTEE_FOLDER = (
    Path.home() / "Desktop" / "AI_Investment_Committee"
)

# Hard cap on total characters sent to the LLM (~40k chars ≈ ~10k tokens).
DEFAULT_MAX_CHARS = 40_000


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract and return all text from a single PDF file.

    Returns an empty string and prints a warning on failure (encrypted,
    corrupted, or unreadable PDFs are silently skipped so the rest of the
    pipeline can continue).
    """
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n".join(pages_text)
    except Exception as exc:
        print(f"  [WARNING] Could not read '{pdf_path.name}': {exc}")
        return ""


def load_all_pdfs(
    folder: Path = INVESTMENT_COMMITTEE_FOLDER,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """Scan *folder* for PDF files and return their combined text.

    - Searches for *.pdf (case-insensitive on all platforms).
    - Concatenates text from every PDF with a filename separator.
    - Truncates the total to *max_chars* to avoid exceeding LLM token limits.
    - Returns an empty string (with a warning) if no PDFs are found or if
      pypdf is not installed.
    """
    if pypdf is None:
        print(
            "[WARNING] 'pypdf' is not installed. PDF context will be skipped.\n"
            "          Fix: pip install pypdf"
        )
        return ""

    if not folder.exists():
        print(
            f"[WARNING] Folder not found: {folder}\n"
            f"          PDF context will be skipped."
        )
        return ""

    # Collect PDFs in a stable order (alphabetical by name).
    pdf_files = sorted(folder.glob("*.pdf")) + sorted(
        p for p in folder.glob("*.PDF") if p not in sorted(folder.glob("*.pdf"))
    )

    if not pdf_files:
        print(
            f"[WARNING] No PDF files found in: {folder}\n"
            f"          PDF context will be skipped. The analysis will rely on\n"
            f"          the hardcoded quality-investing principles instead."
        )
        return ""

    print(f"  Loading {len(pdf_files)} PDF(s) from {folder.name}/...")
    sections: list[str] = []

    for pdf_path in pdf_files:
        print(f"    Reading: {pdf_path.name}")
        text = extract_pdf_text(pdf_path)
        if text:
            sections.append(f"--- Source: {pdf_path.name} ---\n{text}")

    if not sections:
        print("[WARNING] All PDFs were unreadable. PDF context will be skipped.")
        return ""

    combined = "\n\n".join(sections)

    if len(combined) > max_chars:
        combined = (
            combined[:max_chars]
            + "\n\n[...content truncated to fit within token budget]"
        )
        print(
            f"  PDF text truncated to {max_chars:,} characters "
            f"to stay within the LLM token limit."
        )

    print(f"  PDF context loaded: {len(combined):,} characters.\n")
    return combined
