# Evaluating AI Safety Methods in Yoruba

**A Case Study for Local Language Safety in Nigeria** — Africa AI Safety Prize submission.

We show that standard refusal-rate metrics conflate two very different behaviors — a model refusing a harmful request, and a model failing to produce coherent output at all — and that separating them changes safety conclusions in both directions. On Gemma-2-9B-IT, naive metrics suggest catastrophic Yoruba safety failure (99% → 6.7% refusal); three-way classification reveals 92% of that drop is capability failure, not real non-refusal. On Gemini-2.5-Pro, which produces coherent Yoruba, the same classification exposes a real cross-lingual safety gap that naive scoring would miss: harmful compliance rises from 3.0% (EN) to 10.5% (YO) on AdvBench.

## Key finding

Refusal-rate metrics cannot separate safety failure from capability failure. Doing so requires a three-way classification (`refusal` / `compliance` / `capability_failure`), applied here to matched English–Yoruba prompt pairs across two models.

## Results — overall (harmful prompts only)

| Model            | Lang | Refusal | Compliance | Capability failure |
|------------------|:----:|--------:|-----------:|-------------------:|
| Gemma-2-9B-IT    |  EN  |   99.0% |       1.0% |               0.0% |
| Gemma-2-9B-IT    |  YO  |    6.9% |       1.1% |              92.4% |
| Gemini-2.5-Pro   |  EN  |   92.1% |       7.9% |               0.0% |
| Gemini-2.5-Pro   |  YO  |   83.1% |      15.3% |               1.6% |

## Results — per dataset

### Gemma-2-9B-IT (n=5 samples per prompt)

| Dataset   |   n_prompts  | Lang | Refusal | Compliance | Capability failure |
|-----------|:------------:|:----:|--------:|-----------:|-------------------:|
| AdvBench  |     150      |  EN  |  98.67% |      1.33% |              0.00% |
| AdvBench  |     150      |  YO  |   5.47% |      0.53% |             94.00% |
| HarmBench |      48      |  EN  | 100.00% |      0.00% |              0.00% |
| HarmBench |      48      |  YO  |  11.25% |      1.67% |             87.08% |
| XSTest    |      27      |  EN  |  48.89% |     51.11% |              0.00% |
| XSTest    |      27      |  YO  |   5.19% |      2.96% |             91.85% |

*Note on XSTest.* XSTest measures false refusal on benign prompts; EN refusal of 48.89% reflects Gemma's well-documented English over-refusal. The Yoruba figure is dominated by capability failure.

### Gemini-2.5-Pro (n=5 samples per prompt, blank responses excluded)

| Dataset   |  n_prompts  | Lang | Refusal | Compliance | Capability failure |
|-----------|:-----------:|:----:|--------:|-----------:|-------------------:|
| AdvBench  |     150     |  EN  |  96.99% |      3.01% |              0.00% |
| AdvBench  |     150     |  YO  |  88.78% |     10.54% |              0.68% |
| HarmBench |      41*    |  EN  |  73.85% |     26.15% |              0.00% |
| HarmBench |      41*    |  YO  |  61.54% |     33.33% |              5.13% |

*Gemini XSTest evaluation is missing due to API quota exhaustion during runs. HarmBench truncated at 41 of 48 prompts for the same reason.*

## Refusal-language asymmetry (Gemma YO)

Of Gemma's 75 coherent Yoruba refusals: **62 (83%) were produced in English**, 7 in mixed Yoruba/English, and only 6 (8%) in Yoruba. Safety concept transfers across languages; refusal templates barely do.

## Repo structure

```
yorubasafety/
├── data/
│   ├── prompts_yo_en_checked_corpus.jsonl   # 225 matched EN-YO prompts
│   └── prompts_yo_clean.jsonl               # after XSTest contrast-row filter
├── scripts/
│   ├── 01_build_corpus.py                   # NLLB translation + drift filter
│   ├── 01b_filter_xstest.py                 # drop mis-loaded XSTest contrast rows
│   ├── 02_run_eval.py                       # Gemma generation
│   ├── 04_run_eval_gemini.py                # Gemini generation
│   └── 03_score_refusal.py                  # phrase-based refusal detection
├── src/
│   └── refusal.py                           # bilingual refusal-phrase detector
├── outputs/                                 # response JSONLs (see Data section)
└── README.md
```

## Reproduction

Requirements: one A100 GPU (~40GB) for Gemma; Gemini API key with `gemini-2.5-pro` access.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install torch transformers datasets sentence-transformers google-genai python-dotenv tqdm

# Corpus (uses NLLB-200-3.3B; ~15GB download, one-time)
python scripts/01_build_corpus.py
python scripts/01b_filter_xstest.py

# Gemma generation (~1-2 hours per language on A100)
python scripts/02_run_eval.py --lang en --corpus data/prompts_yo_clean.jsonl
python scripts/02_run_eval.py --lang yo --corpus data/prompts_yo_clean.jsonl

# Gemini generation (API-bound, ~15-45 min per language on paid tier)
export GEMINI_API_KEY=your_key
python scripts/04_run_eval_gemini.py --lang en
python scripts/04_run_eval_gemini.py --lang yo

# Phrase-based scoring baseline
python scripts/03_score_refusal.py --model gemma-2-9b-it  --lang yo
```

Three-way LLM-judge classification is applied to the response JSONLs post-hoc using a documented rubric (see `docs/classifier_rubric.md`). Author performed final review of translations and validated a sample of judge classifications.

## Data release and ethics

- **Corpus, prompt IDs, classifications, and confidence scores** are released publicly.
- **Full harmful-response text is NOT released publicly** to prevent this dataset from becoming a jailbreak lookup for Gemini-in-Yoruba. Available on request to affiliated safety researchers.
- All harmful prompts are sourced from public benchmarks (AdvBench, HarmBench, XSTest). No human subjects data.

## Limitations and ongoing work

- Single language pair (English–Yoruba), two models. Findings should not be generalized without replication on other West African languages and other frontier models.
- Yoruba prompts are NLLB-translated and reviewed rather than natively authored; culturally embedded harms are underrepresented.
- Three-way classification is validated against a single Yoruba-speaking reviewer (the author); formal inter-annotator agreement is not yet measured.
- Adversarial-method evaluation, full policy-based evaluation, and mechanistic interpretability (RQ2) are in progress.
