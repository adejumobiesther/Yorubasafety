"""
scripts/02_run_eval.py

Run benchmark-method generation on Gemma-2-9B-IT.

Reads the corpus produced by 01_build_corpus.py, generates n samples per prompt
in either English or Yoruba, and writes one JSONL row per generation.

Usage from project root:
    python scripts/02_run_eval.py --lang en --limit 5     # smoke test
    python scripts/02_run_eval.py --lang en               # full English
    python scripts/02_run_eval.py --lang yo               # full Yoruba
    python scripts/02_run_eval.py --lang yo --dry-run     # verify corpus without model

Outputs:
    outputs/responses_gemma-2-9b-it_{lang}_{method}.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "google/gemma-2-9b-it"
MODEL_SHORT = "gemma-2-9b-it"

DEFAULTS = {
    "n_samples": 5,
    "temperature": 0.7,
    "max_new_tokens": 256,
    "top_p": 0.95,
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
                 f"Run scripts/01_build_corpus.py first.")
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_prompt(record: dict, lang: str) -> tuple[Optional[str], Optional[str]]:
    """Return (prompt_text, source_field_name) for the requested language."""
    if lang == "en":
        text = (record.get("prompt_en") or "").strip()
        return (text, "prompt_en") if text else (None, None)

    if lang == "yo":
        native = (record.get("prompt_yo_native") or "").strip()
        if native:
            return native, "prompt_yo_native"
        mt = (record.get("prompt_yo_mt") or "").strip()
        if mt:
            return mt, "prompt_yo_mt"
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
# Model loading + generation
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    print(f"Loading {model_name} (~18GB in bf16, will take a few minutes on first run)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "gated" in msg or "access" in msg:
            sys.exit(
                "\nERROR: Gemma-2-9B-IT is a gated model. To use it:\n"
                "  1. Accept terms at https://huggingface.co/google/gemma-2-9b-it\n"
                "  2. Run `huggingface-cli login` with a token that has access\n"
            )
        raise

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    model.eval()
    print(f"  loaded on {device}, dtype={dtype}")
    return model, tokenizer


def get_seed(prompt_id: str, base_seed: int) -> int:
    """Deterministic per-prompt seed for reproducibility across reruns."""
    h = hashlib.md5(f"{prompt_id}_{base_seed}".encode()).hexdigest()
    return int(h[:8], 16)


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    device: str,
) -> list[str]:
    """Generate n samples for a single prompt using num_return_sequences."""
    messages = [{"role": "user", "content": prompt}]

    # Two-step: format via chat template, then tokenize separately.
    # (apply_chat_template with tokenize=True returns BatchEncoding in some
    # transformers versions, which model.generate can't unpack.)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_return_sequences=n_samples,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = input_ids.shape[1]
    generated = outputs[:, input_len:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [t.strip() for t in texts]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", choices=VALID_LANGS, required=True,
                        help="Language to evaluate")
    parser.add_argument("--method", choices=VALID_METHODS, default="benchmark",
                        help="Eval method (only 'benchmark' for now)")
    parser.add_argument("--corpus", type=Path,
                        default=DATA_DIR / "prompts_yo_mt.jsonl",
                        help="Path to corpus JSONL")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--n-samples", type=int, default=DEFAULTS["n_samples"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    parser.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    parser.add_argument("--base-seed", type=int, default=DEFAULTS["base_seed"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N prompts (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without loading model or generating")
    args = parser.parse_args()

    # --- Load corpus and filter ---
    records = read_corpus(args.corpus)
    print(f"Loaded {len(records)} records from {args.corpus}")

    usable: list[tuple[dict, str, str]] = []
    skipped_no_prompt = 0
    for rec in records:
        prompt_text, source_field = extract_prompt(rec, args.lang)
        if prompt_text is None:
            skipped_no_prompt += 1
            continue
        usable.append((rec, prompt_text, source_field))

    if skipped_no_prompt:
        print(f"  skipped {skipped_no_prompt} records with no valid {args.lang} prompt")

    if args.limit:
        usable = usable[: args.limit]
        print(f"  --limit applied: processing first {len(usable)} prompts")

    print(f"To process: {len(usable)} prompts × {args.n_samples} samples "
          f"= {len(usable) * args.n_samples} generations")

    # --- Output path + resume state ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"responses_{MODEL_SHORT}_{args.lang}_{args.method}.jsonl"

    done_counts = load_done_counts(output_path)
    to_do = [
        (rec, prompt, src) for rec, prompt, src in usable
        if done_counts.get(rec["id"], 0) < args.n_samples
    ]
    n_already_done = len(usable) - len(to_do)
    if n_already_done:
        print(f"Resuming: {n_already_done} prompts already complete, "
              f"{len(to_do)} remaining")

    print(f"Output: {output_path}")

    if args.dry_run:
        print("\n--dry-run set, exiting before model load")
        if to_do:
            print(f"\nFirst prompt to process:")
            rec, prompt, src = to_do[0]
            print(f"  id={rec['id']}, source_field={src}")
            print(f"  text: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        return

    if not to_do:
        print("Nothing to do. Exiting.")
        return

    # --- Load model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. Generation will be extremely slow.")
    model, tokenizer = load_model(MODEL_NAME, device)

    # --- Generate ---
    gen_metadata = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "base_seed": args.base_seed,
    }

    n_errors = 0
    n_written = 0
    start = time.time()

    for rec, prompt, source_field in tqdm(to_do, desc=f"generating ({args.lang}/{args.method})"):
        prompt_id = rec["id"]
        seed = get_seed(prompt_id, args.base_seed)
        try:
            responses = generate_samples(
                model, tokenizer, prompt,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                seed=seed,
                device=device,
            )
        except torch.cuda.OutOfMemoryError:
            n_errors += 1
            print(f"\n  OOM on {prompt_id}, skipping. Consider reducing max_new_tokens.")
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            n_errors += 1
            print(f"\n  ERROR on {prompt_id}:")
            traceback.print_exc()
            if n_errors >= 3 and n_written == 0:
                print("\n3 errors before any successful generation — aborting.")
                sys.exit(1)
            continue

        for sample_idx, response in enumerate(responses):
            row = {
                "prompt_id": prompt_id,
                "model": MODEL_SHORT,
                "language": args.lang,
                "method": args.method,
                "attack_template": None,
                "sample_idx": sample_idx,
                "prompt_used": prompt,
                "prompt_source_field": source_field,
                "response": response,
                "source_dataset": rec.get("source_dataset"),
                "category": rec.get("category"),
                "subcategory": rec.get("subcategory"),
                "metadata": gen_metadata,
            }
            append_jsonl(row, output_path)
            n_written += 1

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  generations written: {n_written}")
    print(f"  errors: {n_errors}")
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()