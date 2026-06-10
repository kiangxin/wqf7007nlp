import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dev"))

from evaluate import clean_text, get_device

# Sentiment label mappings for OUR evaluation
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

# Original model's actual label mapping
BASE_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

ASPECT_CATEGORIES = [
    "account-management.account-access",
    "company-brand.competitor",
    "company-brand.general-satisfaction",
    "company-brand.reviews",
    "logistics-rides.speed",
    "online-experience.app-website",
    "purchase-booking-experience.ease-of-use",
    "staff-support.attitude-of-staff",
    "staff-support.email",
    "staff-support.phone",
    "value.discounts-promotions",
    "value.price-value-for-money",
]

def predict_base(model, tokenizer, texts, aspects, device, batch_size=32, max_length=256):
    model.eval()
    
    # Tokenize pairwise inputs
    inputs = tokenizer(
        text=texts,
        text_pair=aspects,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    dataset = TensorDataset(
        inputs["input_ids"],
        inputs["attention_mask"]
    )
    
    # Check for token_type_ids (not all tokenizers use it)
    if "token_type_ids" in inputs:
        dataset = TensorDataset(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"]
        )
        use_token_type = True
    else:
        use_token_type = False
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            if use_token_type:
                b_ids, b_mask, b_type = [x.to(device) for x in batch]
                outputs = model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type)
            else:
                b_ids, b_mask = [x.to(device) for x in batch]
                outputs = model(input_ids=b_ids, attention_mask=b_mask)
                
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            
            # Map original model prediction index (e.g. 0=Negative, 1=Neutral, 2=Positive)
            # to our evaluation format (0=Positive, 1=Negative, 2=Neutral)
            mapped_preds = [LABEL2ID[BASE_ID2LABEL[p]] for p in batch_preds]
            preds.extend(mapped_preds)
            
    return preds

def main():
    parser = argparse.ArgumentParser(description="Evaluate raw baseline ABSA model (no fine-tuning)")
    parser.add_argument("--base_model", default="yangheng/deberta-v3-base-absa-v1.1")
    parser.add_argument("--test_csv", default=str(PROJECT_ROOT / "dev" / "data" / "FABSA_test_preprocessed.csv"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading preprocessed test set: {args.test_csv} ...")
    
    # Load test data
    pair_df = pd.read_csv(args.test_csv)
    pair_df["label"] = pair_df["sentiment"].map(LABEL2ID)
    pair_df = pair_df.dropna(subset=["label"]).reset_index(drop=True)
    pair_df["label"] = pair_df["label"].astype(int)
    
    print(f"  Test pairs: {len(pair_df):,}")
    print(f"Loading raw baseline ABSA model (No LoRA fine-tuning): {args.base_model} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    
    # Load model configuration without overriding label mappings dynamically
    # (otherwise it would scramble/reset the classifier weights)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model
    )
    model.to(device)

    
    print("Running predictions...")
    preds = predict_base(
        model, tokenizer,
        pair_df["text"].tolist(),
        pair_df["aspect"].tolist(),
        device, args.batch_size, args.max_length
    )
    
    labels = pair_df["label"].to_numpy()
    
    print("=" * 60)
    print("Base Model (Pre-trained ABSA, No Fine-Tuning) Results")
    print("=" * 60)
    print(classification_report(
        labels, preds,
        target_names=list(LABEL2ID.keys()),
        zero_division=0,
    ))
    
    print(f"Micro F1   : {f1_score(labels, preds, average='micro', zero_division=0):.4f}")
    print(f"Macro F1   : {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")

    # ── Per-aspect breakdown ─────────────────────────────────────────────────
    pair_df["pred"] = preds
    print("\n" + "=" * 60)
    print("Per-aspect macro F1")
    print("=" * 60)
    for aspect_code in ASPECT_CATEGORIES:
        sub = pair_df[pair_df["aspect_code"] == aspect_code]
        if len(sub) == 0:
            continue
        f1 = f1_score(sub["label"], sub["pred"], average="macro", zero_division=0)
        print(f"  {aspect_code:<45}  n={len(sub):>4}  macro-F1={f1:.3f}")

if __name__ == "__main__":
    main()
