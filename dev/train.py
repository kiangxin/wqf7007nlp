"""
Fine-tune yangheng/deberta-v3-base-absa-v1.1 on FABSA for
Aspect Category Sentiment Classification (ACSC).

Input CSV columns used:
  text       – cleaned review text   (primary input)
  aspect     – aspect description    (text_pair)
  sentiment  – positive / negative / neutral  (label)

Usage:
  python train.py
  python train.py --data FABSA_train_preprocessed.csv --epochs 5 --batch_size 16
"""

import argparse
import gc
import json
import os
import warnings
from pathlib import Path

# Must be set before torch is imported so MPS sees it.
# Removes the artificial ~94 % watermark cap; lets the OS manage pressure instead.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent  # → ml_dev/

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "yangheng/deberta-v3-base-absa-v1.1"
DATA_PATH   = str(_HERE / "data" / "FABSA_train_preprocessed.csv")
VAL_PATH    = str(_HERE / "data" / "FABSA_val_preprocessed.csv")   # official FABSA validation split
OUTPUT_DIR  = str(_HERE / "model" / "deberta_absa_finetuned_v2")
MAX_LENGTH  = 256   # covers ~93 % of FABSA pairs; MPS-safe. Raise to 256/512 on GPU
BATCH_SIZE  = 2     # small physical batch for MPS; grad accumulation keeps effective=16
GRAD_ACCUM  = 8     # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
EPOCHS      = 5
LR          = 2e-5
WEIGHT_DECAY= 0.01
WARMUP_RATIO= 0.1
VAL_SPLIT   = 0.1   # fallback fraction used only when --val_data is not available
SEED        = 42

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

# LoRA — only query_proj + value_proj in DeBERTa-v3 attention are adapted;
# the classifier head is trained in full as usual.
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ["query_proj", "value_proj"]


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "aspect", "sentiment"])
    df = df[df["sentiment"].isin(LABEL2ID)].copy()
    df["label"] = df["sentiment"].map(LABEL2ID)
    return df[["text", "aspect", "label"]].reset_index(drop=True)


def load_data(train_path: str, val_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = _clean_df(pd.read_csv(train_path))

    if val_path and Path(val_path).exists():
        val_df = _clean_df(pd.read_csv(val_path))
        print(f"Using official validation split: {val_path}")
    else:
        if val_path:
            print(f"  ⚠  {val_path} not found — falling back to {VAL_SPLIT:.0%} train split")
        train_df, val_df = train_test_split(
            train_df,
            test_size=VAL_SPLIT,
            stratify=train_df["label"],
            random_state=SEED,
        )
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)

    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")
    print(f"Label distribution (train):\n{train_df['label'].map(ID2LABEL).value_counts()}\n")
    print(f"Label distribution (val):\n{val_df['label'].map(ID2LABEL).value_counts()}\n")
    return train_df, val_df


def make_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df)

    def tokenize(batch):
        encoded = tokenizer(
            batch["text"],
            text_pair=batch["aspect"],
            truncation=True,
            max_length=max_length,
            padding=False,          # DataCollatorWithPadding handles dynamic padding
        )
        encoded["labels"] = batch["label"]
        return encoded

    ds = ds.map(tokenize, batched=True, remove_columns=["text", "aspect", "label"])
    ds.set_format("torch")
    return ds



# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy":     accuracy_score(labels, preds),
        "macro_f1":     f1_score(labels, preds, average="macro",    zero_division=0),
        "weighted_f1":  f1_score(labels, preds, average="weighted", zero_division=0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint info callback
# ──────────────────────────────────────────────────────────────────────────────
class CheckpointInfoCallback(TrainerCallback):
    """
    Writes checkpoint_info.json inside each saved checkpoint folder so you
    can tell epoch / macro_f1 at a glance without re-running evaluation.

    Example file contents:
      {
        "global_step": 3416,
        "epoch": 4.0,
        "eval_loss": 0.412,
        "eval_macro_f1": 0.741,
        "eval_accuracy": 0.823
      }
    """

    def __init__(self):
        self._latest_metrics: dict = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._latest_metrics = dict(metrics)

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.exists():
            info = {
                "global_step": state.global_step,
                "epoch":       round(state.epoch or 0, 3),
                **self._latest_metrics,
            }
            (ckpt_dir / "checkpoint_info.json").write_text(
                json.dumps(info, indent=2), encoding="utf-8"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main(args):
    device = get_device()
    print(f"Device: {device}\n")

    # Free any leftover MPS allocations from a previous crashed run
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_df, val_df = load_data(args.data, args.val_data)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # use_fast=False: the fast tokenizer incorrectly tries to parse DeBERTa-v3's
    # SentencePiece .spm file as a tiktoken BPE file, causing a ValueError.
    # Requires: pip install sentencepiece
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_ds = make_hf_dataset(train_df, tokenizer, args.max_length)
    val_ds   = make_hf_dataset(val_df,   tokenizer, args.max_length)

    # ── Model ─────────────────────────────────────────────────────────────────
    # ignore_mismatched_sizes=True re-initialises the classifier head so our
    # label mapping (positive=0, negative=1, neutral=2) is applied cleanly.
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    # Required so gradient checkpointing works with frozen base weights
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Training arguments ────────────────────────────────────────────────────
    use_fp16 = (device == "cuda")
    use_bf16 = (device == "mps")   # MPS works better with bf16 than fp16

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        label_smoothing_factor=0.1,
        seed=SEED,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            CheckpointInfoCallback(),
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Starting fine-tuning …")
    print("=" * 60)
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved → {args.output_dir}")

    # ── Final evaluation report ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Validation set — classification report")
    print("=" * 60)
    pred_output = trainer.predict(val_ds)
    preds  = np.argmax(pred_output.predictions, axis=1)
    labels = pred_output.label_ids
    print(classification_report(labels, preds, target_names=list(LABEL2ID.keys()), zero_division=0))


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default=DATA_PATH,  help="Path to preprocessed train CSV")
    parser.add_argument("--val_data",   default=VAL_PATH,   help="Path to preprocessed val CSV (falls back to train split if missing)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Where to save the model")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--max_length", type=int,   default=MAX_LENGTH)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
