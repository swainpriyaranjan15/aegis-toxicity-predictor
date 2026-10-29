import os
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_CSV,
    OUTPUTS_DIR,
    SPLITS_DIR,
    LABEL_COLUMNS,
    SEED
)

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """
    Minimal cleaning to preserve rich context as advised by the report:
    - Replace URLs with the token 'url'
    - Collapse repeated whitespace / newlines
    - Strip leading and trailing whitespace
    Preserves emojis, punctuation, and capitalization (uncased tokenizer handles lowercasing).
    """
    if not isinstance(text, str):
        return ""
    # Replace URLs
    text = URL_REGEX.sub("url", text)
    # Collapse multiple whitespaces/newlines to single space
    text = WHITESPACE_REGEX.sub(" ", text)
    return text.strip()

def prepare_data(data_path: Path = TRAIN_CSV):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from {data_path}...")
    df = pd.read_csv(data_path)
    total_raw = len(df)
    print(f"Total raw rows: {total_raw}")

    # Drop rows with missing comment_text
    df = df.dropna(subset=["comment_text"]).copy()
    rows_after_na = len(df)

    # Drop exact-duplicate comment_text
    df = df.drop_duplicates(subset=["comment_text"]).copy()
    rows_after_dedup = len(df)
    print(f"Rows dropped due to NA: {total_raw - rows_after_na}, duplicate comment_text dropped: {rows_after_na - rows_after_dedup}")
    print(f"Remaining unique clean rows: {rows_after_dedup}")

    # Clean text
    df["clean_text"] = df["comment_text"].apply(clean_text)
    # Drop any that became empty after cleaning
    df = df[df["clean_text"].str.len() > 0].copy().reset_index(drop=True)
    total_valid = len(df)
    print(f"Rows with non-empty cleaned text: {total_valid}")

    # Compute label statistics
    label_counts = {label: int(df[label].sum()) for label in LABEL_COLUMNS}
    label_percentages = {label: round(float(label_counts[label] / total_valid * 100), 2) for label in LABEL_COLUMNS}

    sum_labels = df[LABEL_COLUMNS].sum(axis=1)
    zero_label_rows = int((sum_labels == 0).sum())
    two_plus_label_rows = int((sum_labels >= 2).sum())
    multi_label_percentage = round(float(two_plus_label_rows / total_valid * 100), 2)
    zero_label_percentage = round(float(zero_label_rows / total_valid * 100), 2)

    stats = {
        "total_rows_raw": total_raw,
        "total_rows_cleaned": total_valid,
        "zero_label_count": zero_label_rows,
        "zero_label_percentage": zero_label_percentage,
        "two_plus_label_count": two_plus_label_rows,
        "two_plus_label_percentage": multi_label_percentage,
        "label_counts": label_counts,
        "label_percentages": label_percentages
    }

    stats_file = OUTPUTS_DIR / "label_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved label stats to {stats_file}")

    # Generate label-frequency bar chart (Outputs/fig_label_frequency.png)
    plt.figure(figsize=(9, 5))
    bars = plt.bar(LABEL_COLUMNS, [label_counts[col] for col in LABEL_COLUMNS], color="#3b82f6", edgecolor="#1d4ed8")
    plt.title("Jigsaw Toxic Comment Dataset - Positive Class Frequencies", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Toxicity Label", fontsize=11, fontweight="medium")
    plt.ylabel("Number of Positive Examples", fontsize=11, fontweight="medium")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / total_valid) * 100
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (total_valid * 0.005), f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, max(label_counts.values()) * 1.15)
    plt.tight_layout()
    fig_freq_file = OUTPUTS_DIR / "fig_label_frequency.png"
    plt.savefig(fig_freq_file, dpi=300)
    plt.close()
    print(f"Saved label frequency chart to {fig_freq_file}")

    # Stratified Split (80% train / 10% val / 10% test)
    # Stratify by combined label signature
    df["strat_key"] = df[LABEL_COLUMNS].astype(str).agg("-".join, axis=1)
    key_counts = df["strat_key"].value_counts()
    rare_keys = key_counts[key_counts < 10].index
    df["strat_group"] = df["strat_key"].apply(lambda k: "other" if k in rare_keys else k)

    # First split 80% train, 20% temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=SEED,
        stratify=df["strat_group"]
    )

    # Second split: split temp equally into val (10%) and test (10%)
    temp_key_counts = temp_df["strat_group"].value_counts()
    temp_rare = temp_key_counts[temp_key_counts < 2].index
    temp_df_strat = temp_df["strat_group"].apply(lambda k: "other" if k in temp_rare else k)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df_strat
    )

    # Save split IDs
    train_df[["id"]].to_csv(SPLITS_DIR / "train_ids.csv", index=False)
    val_df[["id"]].to_csv(SPLITS_DIR / "val_ids.csv", index=False)
    test_df[["id"]].to_csv(SPLITS_DIR / "test_ids.csv", index=False)

    print("\n--- Checkpoint: Split Sizes ---")
    print(f"Train size: {len(train_df)} ({len(train_df)/total_valid*100:.1f}%)")
    print(f"Val size:   {len(val_df)} ({len(val_df)/total_valid*100:.1f}%)")
    print(f"Test size:  {len(test_df)} ({len(test_df)/total_valid*100:.1f}%)")

    print("\n--- Checkpoint: Per-label Positives in Splits ---")
    split_summary = []
    for label in LABEL_COLUMNS:
        tr_cnt = int(train_df[label].sum())
        va_cnt = int(val_df[label].sum())
        te_cnt = int(test_df[label].sum())
        assert tr_cnt > 0 and va_cnt > 0 and te_cnt > 0, f"Label {label} missing in a split!"
        split_summary.append({
            "label": label,
            "train_pos": tr_cnt,
            "val_pos": va_cnt,
            "test_pos": te_cnt
        })
        print(f"  {label:<15}: Train={tr_cnt:<6} Val={va_cnt:<5} Test={te_cnt:<5}")

    print("\nAll splits contain every label! Phase 1 checkpoint successfully satisfied.\n")
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_data()
