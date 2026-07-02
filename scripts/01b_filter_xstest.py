"""
scripts/01b_filter_xstest.py

Drop XSTest contrast rows from the corpus. These are prompts loaded as
"benign" by our XSTest loader but which are actually harmful contrast pairs
(intentionally unsafe-looking) — they inflate FRR when treated as benign.

The upstream corpus has `xstest_label_note` populated with a warning for
these rows; we filter on that field.

Reads:   data/prompts_yo_en_checked_corpus.jsonl
Writes:  data/prompts_yo_clean.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path,
                        default=DATA_DIR / "prompts_yo_en_checked_corpus.jsonl")
    parser.add_argument("--output", type=Path,
                        default=DATA_DIR / "prompts_yo_clean.jsonl")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"ERROR: input not found: {args.input}")

    kept = []
    dropped = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            note = (rec.get("xstest_label_note") or "").strip()
            if note:
                dropped.append(rec)
            else:
                kept.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Input:   {args.input}  ({len(kept) + len(dropped)} records)")
    print(f"Output:  {args.output}  ({len(kept)} records)")
    print(f"Dropped: {len(dropped)} rows flagged in xstest_label_note")

    if dropped:
        print(f"\nSample of dropped rows:")
        for rec in dropped[:3]:
            print(f"  {rec['id']}: {rec.get('prompt_en', '')[:80]}")

    print(f"\nKept breakdown:")
    by_cat = Counter(r.get("category") for r in kept)
    by_src = Counter(r.get("source_dataset") for r in kept)
    for cat, cnt in sorted(by_cat.items()):
        print(f"  category={cat}: {cnt}")
    for src, cnt in sorted(by_src.items()):
        print(f"  source={src}: {cnt}")


if __name__ == "__main__":
    main()