"""
Re-extract BUY/HOLD/SELL from report bodies using the fixed regex, rewrite the
header line in each affected markdown, and patch memory store entries whose
`decision` field is UNKNOWN.

Root cause: the original extract_final_trade_recommendation regex missed the
`**...**` markdown-bold wrapper around the decision token.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPORTS_DIR = BASE / "Capstone" / "reports"
MEMORY_PATH = BASE / "Capstone" / "memory_store" / "memories.jsonl"

FIXED_PATTERNS = [
    r"###\s*Final Investment Recommendation\s*:\s*\*?\*?\s*\[?\s*(BUY|HOLD|SELL)\s*\]?",
    r"Final Investment Recommendation\s*:\s*\*?\*?\s*\[?\s*(BUY|HOLD|SELL)\s*\]?",
    r"Final Recommendation\s*:\s*\*?\*?\s*\[?\s*(BUY|HOLD|SELL)\s*\]?",
]

FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z][A-Z0-9\.\-]*)_trade_recommendation_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<ts>\d{8}_\d{6})\.md$"
)

HEADER_LINE_RE = re.compile(r"Final Trade Recommendation:\s*UNKNOWN", re.IGNORECASE)


def extract_decision(text: str) -> str:
    for pat in FIXED_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


def fix_reports() -> dict[tuple[str, str], str]:
    """Returns map of (ticker, date) -> corrected decision for reports we actually fixed."""
    fixed: dict[tuple[str, str], str] = {}
    for p in sorted(REPORTS_DIR.glob("*_trade_recommendation_*.md")):
        body = p.read_text(encoding="utf-8", errors="ignore")
        if "UNKNOWN" not in body:
            continue
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        ticker, date = m.group("ticker"), m.group("date")

        # Skip the FIRST (possibly-UNKNOWN) match by searching from position of the 2nd header
        # Simpler: find all matches and pick the first BUY/HOLD/SELL (i.e. not UNKNOWN)
        candidates = []
        for pat in FIXED_PATTERNS:
            candidates += re.findall(pat, body, flags=re.IGNORECASE)
        real = next((c.upper() for c in candidates if c.upper() in ("BUY", "HOLD", "SELL")), None)
        if real is None:
            print(f"  SKIP (no real decision found): {p.name}")
            continue

        # Patch header: two places — the "Final Trade Recommendation: UNKNOWN" line and
        # the first "### Final Investment Recommendation: **UNKNOWN**" heading.
        new_body = re.sub(
            r"Final Trade Recommendation:\s*UNKNOWN",
            f"Final Trade Recommendation: {real}",
            body,
        )
        new_body = re.sub(
            r"###\s*Final Investment Recommendation:\s*\*\*UNKNOWN\*\*",
            f"### Final Investment Recommendation: **{real}**",
            new_body,
            count=1,
        )
        if new_body != body:
            p.write_text(new_body, encoding="utf-8")
            fixed[(ticker, date)] = real
            print(f"  fixed {p.name} -> {real}")
    return fixed


def fix_memory(corrections: dict[tuple[str, str], str]) -> int:
    if not MEMORY_PATH.exists():
        return 0
    lines = MEMORY_PATH.read_text(encoding="utf-8").splitlines()
    updated = 0
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        key = (rec.get("ticker", ""), rec.get("date", ""))
        needs_fix = rec.get("decision") == "UNKNOWN" or rec.get("recommendation", "").startswith("UNKNOWN")
        if needs_fix:
            # Try corrections from reports first; else re-extract from the recommendation text itself
            real = corrections.get(key) or extract_decision(rec.get("recommendation", ""))
            if real != "UNKNOWN":
                rec["decision"] = real
                rec["recommendation"] = re.sub(r"^UNKNOWN", real, rec["recommendation"], count=1)
                updated += 1
        new_lines.append(json.dumps(rec, ensure_ascii=False))
    MEMORY_PATH.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


def main() -> None:
    print("Scanning reports...")
    fixed = fix_reports()
    print(f"Fixed {len(fixed)} report(s)")
    print("\nPatching memory store...")
    n = fix_memory(fixed)
    print(f"Patched {n} memory entr(y/ies)")


if __name__ == "__main__":
    main()
