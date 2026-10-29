import os
import sys
import time
import json
import re
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CB_DATASET_CSV,
    OUTPUTS_DIR,
    CYBERBULLY_MODEL_DIR,
    PRETRAINED_MODEL_NAME,
    SEED
)

CB_CLASSES = ["not_cyberbullying", "ethnicity/race", "gender/sexual", "religion"]
LABEL2ID = {lbl: i for i, lbl in enumerate(CB_CLASSES)}
ID2LABEL = {i: lbl for i, lbl in enumerate(CB_CLASSES)}

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@\w+")
WHITESPACE_REGEX = re.compile(r"\s+")

def clean_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = URL_REGEX.sub("", text)
    text = MENTION_REGEX.sub("", text)
    text = WHITESPACE_REGEX.sub(" ", text)
    return text.strip()

class CyberbullyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=96):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True
        )
        item = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class DynamicPadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        features = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch]
        padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        if "labels" in batch[0]:
            padded["labels"] = torch.stack([item["labels"] for item in batch])
        return padded

def plot_cb_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if val > cm.max() / 2 else "black")

    ax.set_xticks(range(len(CB_CLASSES)))
    ax.set_yticks(range(len(CB_CLASSES)))
    ax.set_xticklabels(CB_CLASSES, rotation=25, ha="left", fontsize=9)
    ax.set_yticklabels(CB_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title("Dataset 1 Cyberbullying 4-Class Confusion Matrix", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved cyberbullying confusion matrix to {output_path}")

def train_and_eval(smoke=False, epochs=2, batch_size=32, max_len=96, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Cyberbully model on device: {device}")

    if not CB_DATASET_CSV.exists():
        print(f"Dataset {CB_DATASET_CSV} not found! Skipping Phase 8.")
        return

    print(f"Loading {CB_DATASET_CSV}...")
    df = pd.read_csv(CB_DATASET_CSV)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"]).copy()
    df["clean_text"] = df["text"].apply(clean_tweet)
    df = df[df["clean_text"].str.len() > 0].copy()
    df["target"] = df["label"].map(LABEL2ID)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    print(f"Total valid samples: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())

    # Split: 80 / 10 / 10 stratified
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df["target"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["target"])

    if smoke:
        train_df = train_df.iloc[:2000].copy()
        val_df = val_df.iloc[:500].copy()
        epochs = 1

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=len(CB_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.to(device)

    collator = DynamicPadCollator(tokenizer)
    train_ds = CyberbullyDataset(train_df["clean_text"].values, train_df["target"].values, tokenizer, max_len)
    val_ds = CyberbullyDataset(val_df["clean_text"].values, val_df["target"].values, tokenizer, max_len)
    test_ds = CyberbullyDataset(test_df["clean_text"].values, test_df["target"].values, tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collator)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.06), total_steps)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    best_val_f1 = -1.0
    CYBERBULLY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        # Val eval
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch["labels"].numpy())

        _, _, val_macro_f1, _ = precision_recall_fscore_support(val_targets, val_preds, average="macro", zero_division=0)
        print(f"Cyberbully Epoch {epoch}/{epochs} | Val Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            model.save_pretrained(CYBERBULLY_MODEL_DIR)
            tokenizer.save_pretrained(CYBERBULLY_MODEL_DIR)

    # Final evaluation on held-out test split
    print("\nEvaluating Cyberbullying model on held-out test split...")
    model = AutoModelForSequenceClassification.from_pretrained(CYBERBULLY_MODEL_DIR).to(device)
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(batch["labels"].numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    p, r, f1, sup = precision_recall_fscore_support(test_targets, test_preds, average=None, zero_division=0)
    macro_f1 = float(np.mean(f1))

    cb_metrics = {
        "dataset": "cb_multi_labeled_balanced.csv",
        "num_classes": 4,
        "classes": CB_CLASSES,
        "test_accuracy": round(float(test_acc), 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            cls_name: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(sup[i])
            }
            for i, cls_name in enumerate(CB_CLASSES)
        }
    }

    metrics_out = OUTPUTS_DIR / "cyberbully_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(cb_metrics, f, indent=2)
    print(f"Saved cyberbully metrics to {metrics_out}")

    plot_cb_confusion_matrix(test_targets, test_preds, OUTPUTS_DIR / "cyberbully_fig_confusion_matrix.png")

    meta_out = CYBERBULLY_MODEL_DIR / "model_meta.json"
    cb_meta = {
        "model_version": "Cyberbully V1",
        "base_model": PRETRAINED_MODEL_NAME,
        "classes": CB_CLASSES,
        "max_len": max_len,
        "best_val_macro_f1": round(float(best_val_f1), 4),
        "test_macro_f1": round(macro_f1, 4),
        "test_accuracy": round(float(test_acc), 4)
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(cb_meta, f, indent=2)

    print("\nPhase 8 Cyberbullying Model Completed Successfully!")
    print(f"Test Accuracy: {test_acc:.4f} | Test Macro-F1: {macro_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train_and_eval(smoke=args.smoke, epochs=args.epochs, batch_size=args.batch_size)
