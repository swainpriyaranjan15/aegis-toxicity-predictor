import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_CSV,
    SPLITS_DIR,
    OUTPUTS_DIR,
    MODEL_V4_DIR,
    LABEL_COLUMNS,
    NUM_LABELS,
    PRETRAINED_MODEL_NAME,
    MAX_LEN,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    EPOCHS,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_GRAD_NORM,
    DEFAULT_THRESHOLD,
    POS_WEIGHT_MAX_CLIP,
    SEED
)
from src.preprocess import clean_text

class JigsawDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
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
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

class DynamicPadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        features = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch]
        padded = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        if "labels" in batch[0]:
            padded["labels"] = torch.stack([item["labels"] for item in batch])
        return padded

def compute_pos_weights(train_labels: np.ndarray, clip_max: float = POS_WEIGHT_MAX_CLIP):
    """
    Computes pos_weight per label = min(sqrt(neg/pos), clip_max) strictly on train split.
    V4 improves rare-label recall (threat, identity_hate, severe_toxic).
    """
    pos_counts = train_labels.sum(axis=0)
    total_samples = len(train_labels)
    neg_counts = total_samples - pos_counts
    
    weights = []
    print("\n--- Training Split Class Balances & pos_weight Values ---")
    for i, col in enumerate(LABEL_COLUMNS):
        pos = pos_counts[i]
        neg = neg_counts[i]
        ratio = neg / max(pos, 1)
        w = min(np.sqrt(ratio), clip_max)
        weights.append(w)
        print(f"  {col:<15}: Pos={pos:<6} Neg={neg:<6} Ratio={ratio:<6.2f} Sqrt={np.sqrt(ratio):<5.2f} pos_weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32)

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(input_ids)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    binary_preds = (all_preds >= DEFAULT_THRESHOLD).astype(int)
    macro_f1 = f1_score(all_targets, binary_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1, all_preds, all_targets

def run_training(smoke=False, epochs=None, max_rows=None, resume=False, batch_size=None, lr=None):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_tier = f"Tier 1 - {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "Tier 3 - CPU"
    print(f"Using device: {device} ({compute_tier})")

    epochs = epochs if epochs is not None else (1 if smoke else EPOCHS)
    batch_size = batch_size if batch_size is not None else BATCH_SIZE
    learning_rate = lr if lr is not None else LEARNING_RATE

    MODEL_V4_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load splits
    print("Loading prepared dataset splits...")
    df_raw = pd.read_csv(TRAIN_CSV)
    df_raw = df_raw.dropna(subset=["comment_text"]).drop_duplicates(subset=["comment_text"]).copy()
    df_raw["clean_text"] = df_raw["comment_text"].apply(clean_text)
    df_raw = df_raw[df_raw["clean_text"].str.len() > 0].copy()

    # Match IDs with splits
    train_ids = set(pd.read_csv(SPLITS_DIR / "train_ids.csv")["id"])
    val_ids = set(pd.read_csv(SPLITS_DIR / "val_ids.csv")["id"])
    test_ids = set(pd.read_csv(SPLITS_DIR / "test_ids.csv")["id"])

    train_df = df_raw[df_raw["id"].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df_raw[df_raw["id"].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df_raw[df_raw["id"].isin(test_ids)].copy().reset_index(drop=True)

    if smoke:
        print("\n*** SMOKE TEST MODE: 2,000 train rows, 500 val rows, 1 epoch ***\n")
        train_df = train_df.iloc[:2000].copy()
        val_df = val_df.iloc[:500].copy()
    elif max_rows is not None and max_rows < len(train_df):
        print(f"\n*** REDUCED DATA MODE: Subsetting train to {max_rows} rows (preserving rare labels) ***\n")
        # Keep all positives of rare labels
        rare_mask = (train_df["threat"] == 1) | (train_df["identity_hate"] == 1) | (train_df["severe_toxic"] == 1)
        rare_df = train_df[rare_mask]
        remaining_needed = max_rows - len(rare_df)
        if remaining_needed > 0:
            other_df = train_df[~rare_mask].sample(n=remaining_needed, random_state=SEED)
            train_df = pd.concat([rare_df, other_df]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        else:
            train_df = rare_df.iloc[:max_rows].reset_index(drop=True)

    print(f"Final training rows: {len(train_df)}")
    print(f"Validation rows:     {len(val_df)}")
    print(f"Held-out test rows:  {len(test_df)}")

    # Tokenizer & Model
    print(f"Initializing tokenizer and model ({PRETRAINED_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )
    model.to(device)

    # Compute pos_weight from training set
    train_labels = train_df[LABEL_COLUMNS].values
    val_labels = val_df[LABEL_COLUMNS].values
    pos_weights = compute_pos_weights(train_labels)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    # Datasets and DataLoaders
    train_dataset = JigsawDataset(train_df["clean_text"].values, train_labels, tokenizer, MAX_LEN)
    val_dataset = JigsawDataset(val_df["clean_text"].values, val_labels, tokenizer, MAX_LEN)

    collator = DynamicPadCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Optimizer, Scheduler, Scaler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    total_training_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * epochs
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    scaler = torch.amp.GradScaler(device="cuda") if device.type == "cuda" else None

    # Resume handling
    start_epoch = 1
    best_val_macro_f1 = -1.0
    best_epoch = 1
    checkpoint_file = MODEL_V4_DIR / "checkpoint_latest.pt"
    log_file = OUTPUTS_DIR / "training_log.csv"

    if resume and checkpoint_file.exists():
        print(f"Resuming from checkpoint {checkpoint_file}...")
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_macro_f1 = ckpt.get("best_val_macro_f1", -1.0)
        best_epoch = ckpt.get("best_epoch", 1)
        print(f"Resumed at epoch {start_epoch}, previous best val macro-F1: {best_val_macro_f1:.4f}")
    else:
        # Initialize log CSV
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("epoch,step,train_loss,val_loss,val_macro_f1,elapsed_sec\n")

    print(f"\nBeginning training for {epochs} epochs...")
    print(f"Total steps: {total_training_steps}, Warmup steps: {warmup_steps}, Effective batch size: {batch_size * GRAD_ACCUM_STEPS}")

    start_train_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss = loss / GRAD_ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()

            running_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

            if (step + 1) % (200 * GRAD_ACCUM_STEPS) == 0:
                cur_train_loss = running_loss / (step + 1)
                lr_cur = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Train Loss: {cur_train_loss:.4f} | LR: {lr_cur:.2e}")

        epoch_train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_macro_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {epoch}/{epochs} Finished | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f} | Time: {epoch_elapsed:.1f}s")

        # Save to log
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{step_count},{epoch_train_loss:.6f},{val_loss:.6f},{val_macro_f1:.6f},{epoch_elapsed:.2f}\n")

        # Checkpoint latest for resuming
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch
        }, checkpoint_file)

        # Save best model
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            print(f"==> New best model found at epoch {epoch} (Val Macro-F1: {val_macro_f1:.4f})! Saving to {MODEL_V4_DIR}...")
            model.save_pretrained(MODEL_V4_DIR)
            tokenizer.save_pretrained(MODEL_V4_DIR)

    total_training_sec = time.time() - start_train_time
    print(f"\nTraining completed in {total_training_sec/60:.2f} minutes. Best Epoch: {best_epoch} with Val Macro-F1: {best_val_macro_f1:.4f}")

    # Write model_meta.json
    model_meta = {
        "model_version": "Model V4",
        "base_model": PRETRAINED_MODEL_NAME,
        "num_labels": NUM_LABELS,
        "label_order": LABEL_COLUMNS,
        "max_len": MAX_LEN,
        "default_threshold": DEFAULT_THRESHOLD,
        "seed": SEED,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "pos_weights": {col: float(pos_weights[i]) for i, col in enumerate(LABEL_COLUMNS)},
        "hyperparameters": {
            "batch_size": batch_size,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": batch_size * GRAD_ACCUM_STEPS,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "mixed_precision": "fp16" if device.type == "cuda" else "none"
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(float(best_val_macro_f1), 4),
        "total_training_time_sec": round(total_training_sec, 2),
        "training_date": datetime.now().isoformat(),
        "compute_tier": compute_tier,
        "smoke_test": smoke,
        "reduced_data": (max_rows is not None and max_rows < 127656)
    }

    meta_file = MODEL_V4_DIR / "model_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)
    print(f"Saved model metadata to {meta_file}")

    return model_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on Jigsaw 6-label toxic comments (Model V4)")
    parser.add_argument("--smoke", action="store_true", help="Run 2,000 train rows for 1 epoch as a smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap training rows for compute fallback")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    run_training(
        smoke=args.smoke,
        epochs=args.epochs,
        max_rows=args.max_rows,
        resume=args.resume,
        batch_size=args.batch_size,
        lr=args.lr
    )
