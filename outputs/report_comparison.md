# Reported vs. Measured Model Performance Comparison

This comparison grounds the reported metrics in the academic project report against the actual, independently measured results on the held-out test split (15,958 comments) from the fine-tuned Jigsaw Model V4.

| Metric | Reported in Project Report | Measured in Model V4 Test Set | Difference | Notes |
|---|---|---|---|---|
| **Overall Accuracy** | ≈ 0.98 | **0.9794** (Element-wise / Hamming) <br> **0.9087** (Subset / Exact-match) | Matches Element-wise | The report's reported ≈0.98 corresponds to **element-wise Hamming accuracy** across all 6 binary targets. On imbalanced data with ~90% zero-label comments, element-wise accuracy is high. Exact-match subset accuracy (0.9087) reflects stricter all-label correctness. |
| **Macro F1-score** | ≈ 0.87 | **0.6510** (at threshold 0.5) <br> **0.6877** (with val-tuned thresholds) | Measured honest | Model V4 incorporates class-balanced `pos_weight = min(sqrt(neg/pos), 10)` to boost rare-label recall (threat, severe_toxic, identity_hate). |
| **Toxic-label F1** | ≈ 0.82 | **0.8099** | ±0.03 within report | The general `toxic` category reaches high precision and recall on the held-out test split. |
| **Mean ROC-AUC** | Not reported | **0.9903** | Measured new | Demonstrates strong ranking ability across all 6 toxicity heads. |

### Per-Label Measured Breakdown (Threshold = 0.5)

| Label | Precision | Recall | F1-score | Support | ROC-AUC |
|---|---|---|---|---|---|
| **toxic** | 0.7503 | 0.8797 | 0.8099 | 1530 | 0.9848 |
| **severe_toxic** | 0.3492 | 0.8302 | 0.4916 | 159 | 0.9929 |
| **obscene** | 0.7302 | 0.9160 | 0.8126 | 845 | 0.9923 |
| **threat** | 0.4054 | 0.6667 | 0.5042 | 45 | 0.9908 |
| **insult** | 0.6431 | 0.8793 | 0.7429 | 787 | 0.9894 |
| **identity_hate** | 0.4364 | 0.7254 | 0.5450 | 142 | 0.9917 |
