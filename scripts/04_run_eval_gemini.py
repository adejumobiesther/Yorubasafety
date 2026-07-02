"""
scripts/04_run_eval_gemini.py

Run benchmark-method generation via Google Gemini API (gemini-2.5-pro).

Same output schema as 02_run_eval.py so 03_score_refusal.py works unchanged
(the model short-name is different — pass --model to 03).

Design choices:
  - BLOCK_NONE safety settings across all 4 categories, to measure the
    MODEL's refusal behavior rather than the API content filter.
    (Any residual filtering at other layers is recorded via finish_reason.)
  - n samples per prompt via loop (candidate_count > 1 is unreliable on
    many Gemini models).
  - Rate-limit handling: exponential backoff on 429/quota errors.
  - Resumability: skips prompts already having >= n samples in the output.

Requires:
    export GEMINI_API_KEY=your_key_here
    pip install google-genai

Usage from project root:
    python scripts/04_run_eval_gemini.py --lang en --corpus data/prompts_yo_clean.jsonl
    python scripts/04_run_eval_gemini.py --lang yo --corpus data/prompts_yo_clean.jsonl
    python scripts/04_run_eval_gemini.py --lang yo --limit 5           # smoke test
    python scripts/04_run_eval_gemini.py --lang yo --n-samples 3       # save API calls
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "gemini-2.5-pro"
MODEL_SHORT = "gemini-2.5-pro"

DEFAULTS = {
    "n_samples": 5,
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 256,
    "base_seed": 42,
}

VALID_METHODS = ["benchmark"]
VALID_LANGS = ["en", "yo"]


# ---------------------------------------------------------------------------
# Corpus I/O
# ---------------------------------------------------------------------------

def read_corpus(corpus_path: Path) -> list[dict]:
    if not corpus_path.exists():
        sys.exit(f"ERROR: corpus file not found: {corpus_path}\n"
                 f"Run scripts/01b_filter_xstest.py first.")
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_prompt(record: dict, lang: str) -> tuple[Optional[str], Optional[str]]:
    """Return (prompt_text, source_field). Prefers prompt_yo_final > native > mt."""
    if lang == "en":
        text = (record.get("prompt_en") or "").strip()
        return (text, "prompt_en") if text else (None, None)

    if lang == "yo":
        for field in ("prompt_yo_final", "prompt_yo_native", "prompt_yo_mt"):
            val = (record.get(field) or "").strip()
            if val:
                return val, field
        return None, None

    raise ValueError(f"Unknown language: {lang}")


def load_done_counts(output_path: Path) -> Counter:
    counts: Counter = Counter()
    if not output_path.exists():
        return counts
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                counts[rec["prompt_id"]] += 1
            except (json.JSONDecodeError, KeyError):
                continue
    return counts


def append_jsonl(record: dict, output_path: Path) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
load_dotenv()

def get_client_and_safety():
    """Lazy import so `--help` doesn't require the SDK installed."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("ERROR: google-genai not installed. Run: pip install google-genai")

    client = genai.Client()

    safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    return client, types, safety_settings


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "quota", "rate limit", "resource_exhausted"))


def generate_samples(
    client,
    types_module,
    safety_settings,
    prompt: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[tuple[str, str]]:
    """Call Gemini n times for one prompt. Returns list of (text, finish_reason)."""
    config = types_module.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_new_tokens,
        safety_settings=safety_settings,
    )

    results: list[tuple[str, str]] = []
    for i in range(n_samples):
        # Retry loop for rate limits (waits: 4, 8, 16, 32, 64s)
        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=config,
                )
                text = ""
                finish_reason = "UNKNOWN"
                if response.candidates:
                    cand = response.candidates[0]
                    if cand.finish_reason is not None:
                        finish_reason = str(cand.finish_reason).split(".")[-1]
                    if cand.content and cand.content.parts:
                        text = "".join((p.text or "") for p in cand.content.parts)
                results.append((text.strip(), finish_reason))
                break
            except Exception as e:
                if _is_rate_limit_error(e) and attempt < 4:
                    wait = 2 ** (attempt + 2)   # 4, 8, 16, 32
                    print(f"    rate limited, sleeping {wait}s...")
                    time.sleep(wait)
                    continue
                results.append(("", f"error:{type(e).__name__}"))
                break
        else:
            results.append(("", "rate_limit_exceeded"))
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=VALID_LANGS, required=True)
    parser.add_argument("--method", choices=VALID_METHODS, default="benchmark")
    parser.add_argument("--corpus", type=Path,
                        default=DATA_DIR / "prompts_yo_clean.jsonl")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--n-samples", type=int, default=DEFAULTS["n_samples"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N prompts (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without calling the API")
    args = parser.parse_args()

    records = read_corpus(args.corpus)
    print(f"Loaded {len(records)} records from {args.corpus}")

    usable: list[tuple[dict, str, str]] = []
    skipped = 0
    for rec in records:
        text, field = extract_prompt(rec, args.lang)
        if text is None:
            skipped += 1
            continue
        usable.append((rec, text, field))
    if skipped:
        print(f"  skipped {skipped} records with no valid {args.lang} prompt")

    if args.limit:
        usable = usable[: args.limit]
        print(f"  --limit applied: processing first {len(usable)}")

    print(f"To process: {len(usable)} prompts × {args.n_samples} samples "
          f"= {len(usable) * args.n_samples} API calls")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"responses_{MODEL_SHORT}_{args.lang}_{args.method}.jsonl"

    done_counts = load_done_counts(output_path)
    to_do = [t for t in usable if done_counts.get(t[0]["id"], 0) < args.n_samples]
    n_done = len(usable) - len(to_do)
    if n_done:
        print(f"Resuming: {n_done} prompts already complete, {len(to_do)} remaining")

    print(f"Output: {output_path}")

    if args.dry_run:
        print("\n--dry-run set, exiting")
        return

    if not to_do:
        print("Nothing to do.")
        return

    client, types_module, safety_settings = get_client_and_safety()

    gen_metadata = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "safety": "BLOCK_NONE",
    }

    n_errors = 0
    n_written = 0
    start = time.time()

    for rec, prompt, source_field in tqdm(to_do, desc=f"gemini ({args.lang}/{args.method})"):
        prompt_id = rec["id"]
        try:
            results = generate_samples(
                client, types_module, safety_settings, prompt,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            n_errors += 1
            print(f"\n  ERROR on {prompt_id}:")
            traceback.print_exc()
            if n_errors >= 3 and n_written == 0:
                print("\n3 errors before any success — aborting.")
                sys.exit(1)
            continue

        for sample_idx, (response_text, finish_reason) in enumerate(results):
            row = {
                "prompt_id": prompt_id,
                "model": MODEL_SHORT,
                "language": args.lang,
                "method": args.method,
                "attack_template": None,
                "sample_idx": sample_idx,
                "prompt_used": prompt,
                "prompt_source_field": source_field,
                "response": response_text,
                "source_dataset": rec.get("source_dataset"),
                "category": rec.get("category"),
                "subcategory": rec.get("subcategory"),
                "metadata": {**gen_metadata, "finish_reason": finish_reason},
            }
            append_jsonl(row, output_path)
            n_written += 1

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  API calls written: {n_written}")
    print(f"  errors: {n_errors}")
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()