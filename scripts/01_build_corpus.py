"""
scripts/01_build_corpus.py

Build the Yoruba safety evaluation corpus.

Pipeline:
  1. Load English seed prompts:
       - AdvBench (harmful)          — default n=150
       - HarmBench (harmful)         — default n=50, stratified across categories
       - XSTest benign subset (FRR)  — default n=50
  2. Translate EN → YO using NLLB-200
  3. Back-translate YO → EN, compute cosine similarity to original
  4. Flag items below similarity threshold for manual review
  5. Write JSONL outputs:
       data/seeds_en.jsonl         — English seeds with stable IDs
       data/prompts_yo_mt.jsonl    — NLLB translations + back-trans metadata
       data/backtrans_flags.jsonl  — flagged subset for your review

Usage from project root:
    python scripts/01_build_corpus.py
    python scripts/01_build_corpus.py --limit 10                     # smoke test
    python scripts/01_build_corpus.py --nllb-model facebook/nllb-200-distilled-1.3B
    python scripts/01_build_corpus.py --force                        # re-run all stages
    python scripts/01_build_corpus.py --drift-threshold 0.65

Dependencies:
    pip install torch transformers datasets sentence-transformers tqdm
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DEFAULTS = {
    "n_advbench": 150,
    "n_harmbench": 50,
    "n_xstest": 50,
    "nllb_model": "facebook/nllb-200-3.3B",
    "sim_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "drift_threshold": 0.70,
    "batch_size": 8,
    "seed": 42,
}

NLLB_SRC_EN = "eng_Latn"
NLLB_TGT_YO = "yor_Latn"


# ---------------------------------------------------------------------------
# Record schema
# ---------------------------------------------------------------------------

@dataclass
class PromptRecord:
    id: str
    source_dataset: str            # "advbench" | "harmbench" | "xstest"
    category: str                  # "harmful" | "benign"
    subcategory: Optional[str]     # finer-grained label if available
    prompt_en: str
    prompt_yo_mt: Optional[str] = None
    prompt_yo_native: Optional[str] = None   # kept for future use
    back_translation_en: Optional[str] = None
    backtrans_similarity: Optional[float] = None
    flagged_for_review: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Seed dataset loaders
# ---------------------------------------------------------------------------

def load_advbench(n: int, rng: random.Random) -> List[PromptRecord]:
    """AdvBench — harmful instructions from Zou et al. 2023."""
    print(f"Loading AdvBench (target n={n})...")
    ds = load_dataset("walledai/AdvBench", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n]
    records = []
    for i, idx in enumerate(indices, start=1):
        records.append(PromptRecord(
            id=f"advbench_{i:04d}",
            source_dataset="advbench",
            category="harmful",
            subcategory=None,
            prompt_en=ds[idx]["prompt"].strip(),
        ))
    return records


def load_harmbench(n: int, rng: random.Random) -> List[PromptRecord]:
    """HarmBench standard behaviors — stratified sample across categories."""
    print(f"Loading HarmBench (target n={n}, stratified)...")
    ds = load_dataset("walledai/HarmBench", "standard", split="train")

    by_category: dict[str, List[int]] = {}
    for i, row in enumerate(ds):
        cat = row.get("category") or "uncategorized"
        by_category.setdefault(cat, []).append(i)

    per_cat = max(1, n // max(1, len(by_category)))
    indices: List[int] = []
    for cat, idxs in by_category.items():
        rng.shuffle(idxs)
        indices.extend(idxs[:per_cat])

    rng.shuffle(indices)
    indices = indices[:n]

    records = []
    for i, idx in enumerate(indices, start=1):
        row = ds[idx]
        records.append(PromptRecord(
            id=f"harmbench_{i:04d}",
            source_dataset="harmbench",
            category="harmful",
            subcategory=row.get("category"),
            prompt_en=row["prompt"].strip(),
        ))
    return records


def load_xstest_benign(n: int, rng: random.Random) -> List[PromptRecord]:
    """XSTest — safe-but-sounds-unsafe prompts for measuring False Refusal Rate."""
    print(f"Loading XSTest benign (target n={n})...")
    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")

    # XSTest has 'type' field: "safe" rows are benign; "unsafe" are actually harmful.
    safe_rows = [i for i, row in enumerate(ds) if "safe" in row["type"] and "unsafe" not in row["type"]]
    rng.shuffle(safe_rows)
    safe_rows = safe_rows[:n]

    records = []
    for i, idx in enumerate(safe_rows, start=1):
        row = ds[idx]
        records.append(PromptRecord(
            id=f"xstest_{i:04d}",
            source_dataset="xstest",
            category="benign",
            subcategory=row.get("type"),
            prompt_en=row["prompt"].strip(),
        ))
    return records


# ---------------------------------------------------------------------------
# Translation (NLLB)
# ---------------------------------------------------------------------------

def load_nllb(model_name: str, device: str):
    print(f"Loading NLLB: {model_name} (this can take a few minutes the first time)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def translate_batch(
    texts: List[str],
    model,
    tokenizer,
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    device: str,
    desc: str = "translating",
) -> List[str]:
    """Translate a list of strings src_lang → tgt_lang using NLLB."""
    tokenizer.src_lang = src_lang
    bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    results: List[str] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_new_tokens=512,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(decoded)
    return results


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def compute_similarities(
    originals: List[str],
    back_translations: List[str],
    sim_model_name: str,
    device: str,
) -> List[float]:
    """Cosine similarity between original EN and back-translated EN, per item."""
    print(f"Loading similarity model: {sim_model_name}...")
    model = SentenceTransformer(sim_model_name, device=device)

    emb_orig = model.encode(originals, convert_to_tensor=True, show_progress_bar=True)
    emb_back = model.encode(back_translations, convert_to_tensor=True, show_progress_bar=True)

    sims = util.cos_sim(emb_orig, emb_back).diagonal().cpu().tolist()
    return [float(s) for s in sims]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  wrote {path}")


def read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_seed_set(args, rng: random.Random) -> List[PromptRecord]:
    records: List[PromptRecord] = []
    records += load_advbench(args.n_advbench, rng)
    records += load_harmbench(args.n_harmbench, rng)
    records += load_xstest_benign(args.n_xstest, rng)
    if args.limit:
        records = records[: args.limit]
    print(f"Total seed records: {len(records)} "
          f"(harmful={sum(1 for r in records if r.category == 'harmful')}, "
          f"benign={sum(1 for r in records if r.category == 'benign')})")
    return records


def translate_records(records: List[PromptRecord], args, device: str) -> List[PromptRecord]:
    model, tokenizer = load_nllb(args.nllb_model, device)

    en_texts = [r.prompt_en for r in records]

    print("\n[1/2] Forward translation EN → YO")
    yo_texts = translate_batch(
        en_texts, model, tokenizer,
        src_lang=NLLB_SRC_EN, tgt_lang=NLLB_TGT_YO,
        batch_size=args.batch_size, device=device,
        desc="EN→YO",
    )

    print("\n[2/2] Back-translation YO → EN")
    back_en_texts = translate_batch(
        yo_texts, model, tokenizer,
        src_lang=NLLB_TGT_YO, tgt_lang=NLLB_SRC_EN,
        batch_size=args.batch_size, device=device,
        desc="YO→EN",
    )

    for rec, yo, back in zip(records, yo_texts, back_en_texts):
        rec.prompt_yo_mt = yo.strip()
        rec.back_translation_en = back.strip()

    # Free NLLB before loading similarity model
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\nComputing drift similarities...")
    sims = compute_similarities(en_texts, back_en_texts, args.sim_model, device)
    for rec, sim in zip(records, sims):
        rec.backtrans_similarity = round(sim, 4)
        rec.flagged_for_review = sim < args.drift_threshold

    n_flagged = sum(1 for r in records if r.flagged_for_review)
    print(f"Flagged for review (sim < {args.drift_threshold}): {n_flagged} / {len(records)}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-advbench", type=int, default=DEFAULTS["n_advbench"])
    parser.add_argument("--n-harmbench", type=int, default=DEFAULTS["n_harmbench"])
    parser.add_argument("--n-xstest", type=int, default=DEFAULTS["n_xstest"])
    parser.add_argument("--nllb-model", type=str, default=DEFAULTS["nllb_model"])
    parser.add_argument("--sim-model", type=str, default=DEFAULTS["sim_model"])
    parser.add_argument("--drift-threshold", type=float, default=DEFAULTS["drift_threshold"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--limit", type=int, default=None, help="Cap total records (smoke test)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    seeds_path = args.data_dir / "seeds_en.jsonl"
    yo_mt_path = args.data_dir / "prompts_yo_mt.jsonl"
    flags_path = args.data_dir / "backtrans_flags.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    rng = random.Random(args.seed)

    # ---- Stage 1: build seed set ----
    if seeds_path.exists() and not args.force:
        print(f"Seed file exists: {seeds_path} — loading (use --force to rebuild)")
        seed_dicts = read_jsonl(seeds_path)
        records = [PromptRecord(**{k: v for k, v in d.items() if k in PromptRecord.__annotations__})
                   for d in seed_dicts]
    else:
        records = build_seed_set(args, rng)
        write_jsonl([r.to_dict() for r in records], seeds_path)

    # ---- Stage 2: translate + back-translate + flag ----
    if yo_mt_path.exists() and not args.force:
        print(f"Translation file exists: {yo_mt_path} — skipping (use --force to retranslate)")
    else:
        records = translate_records(records, args, device)
        write_jsonl([r.to_dict() for r in records], yo_mt_path)

        flagged = [r.to_dict() for r in records if r.flagged_for_review]
        write_jsonl(flagged, flags_path)

    print("\nDone.")
    print(f"  Seeds:       {seeds_path}")
    print(f"  Yoruba (MT): {yo_mt_path}")
    print(f"  Flagged:     {flags_path}")


if __name__ == "__main__":
    main()