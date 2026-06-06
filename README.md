# FABSA Aspect-Based Sentiment Analysis Pipeline

Two-stage ABSA pipeline on the [jordiclive/FABSA](https://huggingface.co/datasets/jordiclive/FABSA) dataset:

1. **Aspect Category Detection (ACD)** — LLM via OpenRouter identifies which of 12 predefined aspect categories are mentioned in a review, with evidence spans and per-aspect reasoning.
2. **Aspect Category Sentiment Classification (ACSC)** — Fine-tuned `yangheng/deberta-v3-base-absa-v1.1` (LoRA) predicts sentiment polarity (positive / negative / neutral) for each detected aspect.

The best-performing ACD LLM model from evaluation was `google/gemini-3.5-flash` (exact match 45 %, avg reasoning score 7.97 / 10).

---

## Project structure

```
wqf7007nlp/
├── dev/                              # ML development pipeline
│   ├── data/
│   │   ├── FABSA_train_preprocessed.csv
│   │   ├── FABSA_val_preprocessed.csv
│   │   ├── FABSA_test_preprocessed.csv
│   │   ├── FABSA_golden_test_100.csv  # fixed test set for LLM eval
│   │   ├── trustpilot_reviews.json
│   │   └── trustpilot_reviews.csv
│   ├── model/
│   │   └── deberta_absa_finetuned/   # saved LoRA adapter (after training)
│   ├── notebooks/
│   │   ├── preprocessing.ipynb       # clean & format training data
│   │   └── eda.ipynb                 # exploratory analysis
│   ├── train.py                      # fine-tune DeBERTa-v3 with LoRA
│   ├── evaluate.py                   # evaluate on FABSA test split
│   └── inference.py                  # full two-stage pipeline (CLI + batch)
│
├── eval/                             # LLM ACD model selection experiment
│   ├── llm/
│   │   ├── prepare_golden_test.py    # sample 100-review fixed test set from FABSA
│   │   ├── llm_judge_eval.py         # benchmark 3 LLMs: exact match + Claude judge
│   │   ├── export_predictions.py     # export ACD+ABSA predictions for golden test
│   │   └── export_trustpilot.py      # run full pipeline on Trustpilot scrapes
│   └── llm_judge/
│       ├── cache/                    # ACD LLM response cache (per-model subdirs)
│       ├── judge_cache/              # Claude judge score cache
│       ├── results.csv               # per-review eval results for all 3 models
│       └── metadata.json             # run summary + model comparison scores
│
├── app/                              # Web demo
│   ├── server.py                     # FastAPI backend
│   ├── frontend/
│   │   └── index.html                # single-page UI
│   └── data/output/                  # pre-analyzed Trustpilot predictions
│       ├── predictions_trustpilot_banking.json
│       ├── predictions_trustpilot_it.json
│       ├── predictions_trustpilot_ecommerce.json
│       └── predictions_trustpilot_fashion.json
│
├── utils/
│   ├── json_parser.py                # robust multi-pass LLM JSON parser
│   └── _batch_scrape.py              # one-shot Trustpilot batch scraper
│
├── docs/
│   └── QNA.md
├── .env                              # OPENROUTER_API_KEY (not committed)
└── requirements.txt
```

---

## Setup

We recommend using **Python 3.11** for this project. You can set up your environment using either **Conda** or a standard **Python virtual environment (venv)**.

### Option A: Setup with Conda (Recommended)

If you use Anaconda or Miniconda, run the following commands from the project root:

```bash
# 1. Create a conda environment with Python 3.11
conda create -n <your-env-name> python=3.11 -y

# 2. Activate the environment
conda activate <your-env-name>

# 3. Install packages
pip install -r requirements.txt

# 4. Install Playwright browser dependencies (only needed if re-running the scraper)
playwright install chromium
```

### Option B: Setup with Python `venv`

If you prefer using the built-in python `venv` module, run the following commands:

```bash
# 1. Create a virtual environment (e.g., named 'venv') using Python 3.11
python3.11 -m venv venv

# 2. Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install packages
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install Playwright browser dependencies (only needed if re-running the scraper)
playwright install chromium
```

---

Create a `.env` file at the project root:

```
OPENROUTER_API_KEY=sk-or-...
```

---

## Step 1 — Data preprocessing

Open and run all cells in `dev/notebooks/preprocessing.ipynb` from the project root.

The notebook applies the following steps to the raw FABSA training data loaded from HuggingFace:

1. Deduplication on raw text
2. Lexical normalisation via `contractions`
3. Emoji demojization
4. HTML entity unescaping + tag/email removal
5. URL masking → `[URL]` token
6. Repeated punctuation collapse, whitespace strip
7. Label parsing from `label_codes` column
8. DeBERTa-v3 pairwise expansion — one row per `(review, aspect)` pair

Outputs written to `dev/data/`:

| File | Description |
|---|---|
| `FABSA_train_preprocessed.csv` | Training set in pairwise format |
| `FABSA_val_preprocessed.csv` | Validation set (official FABSA split) |
| `FABSA_test_preprocessed.csv` | Test set for ACSC evaluation |

Columns: `id, data_source, industry, text, aspect, aspect_code, sentiment, formatted_input`

---

## Step 2 — Fine-tune DeBERTa-v3 (ACSC)

Fine-tunes `yangheng/deberta-v3-base-absa-v1.1` with LoRA (`r=16, alpha=32`) on `query_proj` and `value_proj` attention layers. Early stopping with patience 2 on validation macro F1.

```bash
cd dev
python train.py
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--data` | `data/FABSA_train_preprocessed.csv` | Preprocessed train CSV |
| `--val_data` | `data/FABSA_val_preprocessed.csv` | Preprocessed val CSV |
| `--output_dir` | `model/deberta_absa_finetuned` | Where to save the LoRA adapter |
| `--epochs` | `5` | Training epochs |
| `--batch_size` | `2` | Per-device batch size (effective batch = 16 with grad accumulation ×8) |
| `--lr` | `2e-5` | Learning rate |
| `--max_length` | `256` | Tokeniser sequence length |

Adapter is saved to `dev/model/deberta_absa_finetuned/`.

---

## Step 3 — Evaluate ACSC model

Loads the official FABSA test split from HuggingFace and evaluates the saved adapter. Reports overall accuracy, macro F1, weighted F1, and per-aspect macro F1.

```bash
cd dev
python evaluate.py
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--adapter` | `model/deberta_absa_finetuned` | Path to saved LoRA adapter |
| `--test_csv` | `data/FABSA_test_preprocessed.csv` | Preprocessed test CSV |
| `--batch_size` | `16` | Inference batch size |
| `--max_length` | `256` | Tokeniser sequence length |

---

## Step 4 — LLM ACD model selection

Benchmarks 3 LLM models (`google/gemini-3.5-flash`, `openai/gpt-4.1-mini`, `qwen/qwen3.6-flash`) on the ACD task using:
- **Exact match** — predicted aspect slug set == ground truth slug set
- **Claude judge** — scores reasoning quality 0–10 per review

### 4a. Prepare the golden test set (run once)

Samples 100 unique, conflict-free reviews from the FABSA test split.

```bash
cd eval
python llm/prepare_golden_test.py
# Output: ../dev/data/FABSA_golden_test_100.csv
```

### 4b. Run the benchmark

```bash
cd eval
python llm/llm_judge_eval.py
```

Options:

| Argument | Default | Description |
|---|---|---|
| `--models` | all 3 | Space-separated OpenRouter model slugs |
| `--n_reviews` | 0 (all) | Limit to first N reviews (quick test) |
| `--skip_judge` | false | Run exact-match only, skip Claude judge |
| `--output_dir` | `llm_judge/` | Where to write `results.csv` and `metadata.json` |

Results are checkpointed after each model to `eval/llm_judge/results.csv`. The final summary in `eval/llm_judge/metadata.json` shows:

| Model | Exact Match | Avg Reasoning Score |
|---|---|---|
| `google/gemini-3.5-flash` | **0.45** | **7.97 / 10** |
| `qwen/qwen3.6-flash` | 0.40 | 7.58 / 10 |
| `openai/gpt-4.1-mini` | 0.36 | 7.42 / 10 |

**→ `google/gemini-3.5-flash` is used as the ACD model throughout.**

---

## Step 5 — Export Trustpilot predictions for the web demo

Runs the full ACD + ACSC pipeline on the scraped Trustpilot reviews and writes prediction files consumed by the web server.

```bash
cd eval
# Export one domain
python llm/export_trustpilot.py --domain banking
python llm/export_trustpilot.py --domain it
python llm/export_trustpilot.py --domain ecommerce
python llm/export_trustpilot.py --domain fashion

# Or all domains at once
python llm/export_trustpilot.py --domain all
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--domain` | `banking` | Domain to export, or `all` |
| `--model` | `google/gemini-3.5-flash` | ACD LLM model slug |
| `--adapter` | `../dev/model/deberta_absa_finetuned` | ACSC adapter path |
| `--n_reviews` | 0 (all) | Limit reviews (0 = full domain) |
| `--min_confidence` | `0.70` | Min LLM confidence to pass an aspect to ACSC |

LLM responses are served from the existing cache in `eval/llm_judge/cache/` — no new API calls are made for reviews already evaluated.

Output files go to `app/data/output/predictions_trustpilot_{domain}.json`.

---

## Step 6 — Run the web demo

```bash
uvicorn app.server:app --reload --port 8501
```

Open `http://localhost:8501`.

**Browse tab** — pre-analyzed Trustpilot reviews grouped by domain (Banking, IT, E-Commerce, Fashion). A domain stats card shows sentiment breakdown (positive / negative / neutral counts) and top detected aspect categories. Individual reviews display aspect chips colour-coded by sentiment (green = positive, red = negative, grey = neutral) with LLM confidence and ACSC confidence scores.

**Live Analysis tab** — paste any review text for real-time ACD + ACSC inference. Optional context fields (company name, industry) prepend business context to the LLM prompt for more accurate aspect detection.

---

## Step 7 — CLI inference (optional)

Run the pipeline on arbitrary text from the command line.

```bash
cd dev

# Single review
python inference.py --text "Great app but delivery took forever"

# Interactive mode
python inference.py

# Batch CSV (must have a 'text' column)
python inference.py --input reviews.csv --output predictions.csv
```

Key options:

| Argument | Default | Description |
|---|---|---|
| `--llm_model` | `google/gemini-3.5-flash` | Any OpenRouter model slug |
| `--adapter` | `model/deberta_absa_finetuned` | Path to saved LoRA adapter |
| `--min_confidence` | `0.70` | Min LLM confidence to run ACSC on an aspect |
| `--cache_dir` | `` | Directory to cache LLM responses across runs |
| `--api_key` | `` | OpenRouter API key (or set `OPENROUTER_API_KEY` in `.env`) |

Example output:

```
  [ 6] online experience: app or website         LLM:98%  → positive   ABSA:95.67%
  [ 5] logistics: speed                           LLM:95%  → negative   ABSA:91.20%
```
