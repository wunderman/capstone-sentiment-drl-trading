"""
Seed the LangGraph EmbeddingMemory store from existing trade-recommendation
reports in Capstone/reports/.

Parses each markdown report for (ticker, date, fundamentals, sentiment,
sentiment_score, final_recommendation) and replays them into memory so the
next LangGraph run has a warm cache.

De-dups by (ticker, date) — keeps the newest timestamp per pair.

Usage:
    python seed_langgraph_memory.py                # seed into default store
    python seed_langgraph_memory.py --dry-run      # print what would seed
    python seed_langgraph_memory.py --reset        # wipe store before seeding
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "Capstone"))

from graph.memory_store import EmbeddingMemory, build_situation_text, DEFAULT_STORE  # noqa: E402


# Inlined to avoid importing trade_generation_pipeline (which triggers FinBERT load).
def extract_final_trade_recommendation(final_text: str) -> str:
    if not final_text:
        return "UNKNOWN"
    patterns = [
        r"###\s*Final Investment Recommendation\s*:\s*\*?\*?\s*\[?\s*(BUY|HOLD|SELL)\s*\]?",
        r"Final Investment Recommendation\s*:\s*\*?\*?\s*\[?\s*(BUY|HOLD|SELL)\s*\]?",
        r"Final Recommendation\s*:\s*\*?\*?\s*\[?\s*(BUY|HOLD|SELL)\s*\]?",
    ]
    for pattern in patterns:
        m = re.search(pattern, final_text, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"

REPORTS_DIR = BASE_DIR / "Capstone" / "reports"
FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Z][A-Z0-9\.\-]*)_trade_recommendation_"
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<ts>\d{8}_\d{6})\.md$"
)


def split_sections(body: str) -> dict[str, str]:
    """Map section-heading → body for both `## H` and `### H` headings."""
    lines = body.splitlines()
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in lines:
        m = re.match(r"^##+\s+(.+?)\s*$", line)
        if m:
            if current is not None:
                sections[current.strip().lower()] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sections[current.strip().lower()] = "\n".join(buf).strip()
    return sections


def parse_sentiment_score(body: str) -> float:
    m = re.search(r"\*\*Sentiment Score:\*\*\s*([-+]?[0-9]*\.?[0-9]+)", body)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return 0.0


def parse_conviction(text: str) -> str:
    m = re.search(r"Conviction Level\s*:\s*\[?\s*(High|Medium|Low)\s*\]?", text, flags=re.IGNORECASE)
    return m.group(1).title() if m else ""


def parse_report(path: Path) -> dict | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    ticker = m.group("ticker")
    date = m.group("date")
    ts = m.group("ts")

    body = path.read_text(encoding="utf-8", errors="ignore")
    sections = split_sections(body)

    fundamentals = sections.get("fundamentals analysis", "")
    sentiment = sections.get("sentiment analysis", "")
    sentiment_score = parse_sentiment_score(body)

    final_block = (
        sections.get("final recommendation")
        or sections.get("final investment recommendation")
        or ""
    )
    if not final_block:
        hit = re.search(
            r"###\s*Final Investment Recommendation.*?(?=^---|\Z)",
            body,
            flags=re.DOTALL | re.MULTILINE,
        )
        final_block = hit.group(0) if hit else ""

    decision = extract_final_trade_recommendation(final_block or body)
    conviction = parse_conviction(final_block or body)

    return {
        "ticker": ticker,
        "date": date,
        "timestamp": ts,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "final_block": final_block.strip() or "Unknown",
        "decision": decision,
        "conviction": conviction,
        "path": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reset", action="store_true", help="Delete existing store before seeding.")
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    args = ap.parse_args()

    store_path = Path(args.store)
    if args.reset and store_path.exists() and not args.dry_run:
        print(f"Wiping existing store: {store_path}")
        store_path.unlink()

    reports = sorted(REPORTS_DIR.glob("*_trade_recommendation_*.md"))
    print(f"Found {len(reports)} trade-recommendation reports in {REPORTS_DIR}")

    parsed: list[dict] = []
    skipped = 0
    for p in reports:
        rec = parse_report(p)
        if rec is None:
            skipped += 1
            continue
        parsed.append(rec)
    print(f"Parsed {len(parsed)}, skipped {skipped} (bad filename)")

    # De-dup by (ticker, date), keep newest timestamp
    newest: dict[tuple[str, str], dict] = {}
    for r in parsed:
        key = (r["ticker"], r["date"])
        if key not in newest or r["timestamp"] > newest[key]["timestamp"]:
            newest[key] = r
    print(f"After dedup by (ticker,date): {len(newest)} unique entries")

    # Load existing store to know what's already seeded
    memory = EmbeddingMemory(store_path=store_path)
    already = {(m.get("ticker", ""), m.get("date", "")) for m in memory.memories}
    print(f"Store already contains {len(already)} entries")

    to_add = [r for k, r in newest.items() if k not in already]
    print(f"Will add {len(to_add)} new entries")

    if args.dry_run:
        for r in to_add[:10]:
            print(f"  would add {r['ticker']} {r['date']} → {r['decision']} (conv={r['conviction']})")
        if len(to_add) > 10:
            print(f"  ... and {len(to_add) - 10} more")
        return

    for r in to_add:
        situation = build_situation_text(
            ticker=r["ticker"],
            date=r["date"],
            fundamentals_report=r["fundamentals"],
            sentiment_report=r["sentiment"],
            sentiment_score=r["sentiment_score"],
        )
        memory.add_memory(
            situation=situation,
            recommendation=f"{r['decision']}\n\n{r['final_block'][:1500]}",
            ticker=r["ticker"],
            date=r["date"],
            conviction=r["conviction"],
            extra={"decision": r["decision"], "seeded_from": Path(r["path"]).name},
        )
    print(f"Seeded {len(to_add)} entries. Store now has {len(memory)} total.")


if __name__ == "__main__":
    main()
