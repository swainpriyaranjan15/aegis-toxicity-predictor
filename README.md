# AI-Powered Cyberbullying & Toxicity Predictor with Contextual Moderation

An end-to-end multi-label Natural Language Processing system for fine-grained toxic language classification and contextual moderation advice, powered by a fine-tuned DistilBERT model (**Model V4**) with class-weighted BCE loss.

---

## 1. System Architecture

```
                                  +--------------------------------------------------+
                                  |                 User / Client                    |
                                  |    (Single-Page HTML5/CSS3/JS Web Interface)     |
                                  +--------------------------------------------------+
                                                           |
                                                HTTP POST /predict (JSON)
                                                           v
                                  +--------------------------------------------------+
                                  |                FastAPI Backend                   |
                                  |    - Input validation (1-2000 chars, non-empty)  |
                                  |    - CORS & Static file hosting                  |
                                  |    - Error handling & Privacy protection         |
                                  +--------------------------------------------------+
                                                           |
                                                           v
                                  +--------------------------------------------------+
                                  |         Inference & Moderation Engine            |
                                  |                  (predict.py)                    |
                                  |  - Minimal context-preserving text cleaning      |
                                  |  - DistilBERT uncased tokenization (max_len=128) |
                                  |  - 6-head sigmoid forward pass                   |
                                  |  - Ordered contextual moderation policy          |
                                  +--------------------------------------------------+
                                          /                                  \
                                         v                                    v
            +-----------------------------------------+     +-----------------------------------------+
            |          Jigsaw 6-Label Model V4        |     |           Moderation Policy             |
            | - toxic          - severe_toxic         |     | - Score & score_percent                 |
            | - obscene        - threat               |     | - Severity: None, Low, Mod, High, Crit  |
            | - insult         - identity_hate        |     | - Action: allow, review, warn, hide,    |
            | (Dynamic pos_weight rare-label recall)  |     |           escalate                      |
            +-----------------------------------------+     +-----------------------------------------+
```

---

## 2. Moderation Policy & Decision Rules

Classification and moderation are strictly separated. Probabilities are generated for each label via independent sigmoids, and the application evaluates rules in strict order:

| Priority | Rule Condition | Severity | Action | Rationale |
|:---:|---|:---:|:---:|---|
| **1** | Any of `severe_toxic`, `threat`, or `identity_hate` detected ($\ge 0.50$) | **Critical** | `escalate` | High-risk, dangerous, or legally sensitive language requiring immediate priority human moderator attention. |
| **2** | $\text{score} \ge 0.80$ OR 2+ labels detected ($\ge 0.50$) | **High** | `hide` | Strong toxicity or multi-vector abusive attack; content hidden from public view to protect the community. |
| **3** | Exactly 1 moderate label detected ($\text{score} \in [0.50, 0.80)$) | **Moderate** | `warn` | Mildly offensive or insulting language; warning notice applied to the user. |
| **4** | No labels detected, but borderline confidence ($\text{score} \in [0.30, 0.50)$) | **Low** | `review` | Context-dependent or ambiguous phrasing queued for asynchronous human review. |
| **5** | Otherwise ($\text{score} < 0.30$) | **None** | `allow` | Safe, benign content permitted immediately. |

---

## 3. Measured Model Performance (Held-out Test Split)

Evaluation was conducted exclusively on the held-out test split of **15,958 comments** (never exposed during training or validation).

- **Hardware Tier:** Tier 1 — NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.8, Mixed Precision FP16, Dynamic Padding).
- **Training Time:** 37.72 minutes (3 Epochs, Best Epoch: 3).
- **Element-wise (Hamming) Accuracy:** **0.9794 (97.94%)** *(Matches reported ~0.98)*
- **Subset (Exact-Match) Accuracy:** **0.9087 (90.87%)**
- **Macro F1-Score:** **0.6510** (default threshold 0.5) / **0.6877** (validation-tuned thresholds)
- **Toxic-label F1-Score:** **0.8099** *(Near parity with reported ~0.82)*
- **Mean ROC-AUC:** **0.9903** across all 6 heads
- **Inference Latency:** Mean: **9.28 ms** | p50: **8.88 ms** | p95: **14.63 ms** (50 requests on RTX 3050)

### Per-Label Performance Breakdown (Held-out Test Split, Threshold = 0.5)

| Toxicity Label | Precision | Recall | F1-Score | Support | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **toxic** | 0.7503 | 0.8797 | **0.8099** | 1,530 | 0.9848 |
| **severe_toxic** | 0.3492 | **0.8302** | 0.4916 | 159 | 0.9929 |
| **obscene** | 0.7302 | **0.9160** | **0.8126** | 845 | 0.9923 |
| **threat** | 0.4054 | **0.6667** | 0.5042 | 45 | 0.9908 |
| **insult** | 0.6431 | **0.8793** | **0.7429** | 787 | 0.9894 |
| **identity_hate** | 0.4364 | **0.7254** | 0.5450 | 142 | 0.9917 |

*Note on Rare-Label Recall:* The `pos_weight = min(sqrt(neg/pos), 10)` balancing mechanism succeeded in elevating recall on rare categories (`severe_toxic` reached 83.0% recall, `identity_hate` reached 72.5% recall, `threat` reached 66.7% recall).

---

## 4. Setup and Installation

### Prerequisites
- Python 3.10+ (Tested on Python 3.14 with CUDA PyTorch)
- NVIDIA GPU recommended (CUDA supported), or CPU fallback

```bash
# Clone the repository
git clone <repo-url>
cd toxicity-predictor

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Training and Evaluation

### Quick Smoke Test (2,000 samples, 1 epoch)
```bash
python src/train.py --smoke
```

### Full Model V4 Training
```bash
python src/train.py
```
*Key features:*
- Uses class-weighted `pos_weight = min(sqrt(neg/pos), 10.0)` computed solely on the training split to boost rare-label recall (`threat`, `severe_toxic`, `identity_hate`).
- Mixed precision (`torch.amp.autocast`) with dynamic padding.
- Saves the best checkpoint based on validation Macro-F1 to `models/jigsaw_v4/`.

### Run Comprehensive Evaluation
```bash
python src/evaluate.py
```
Generates:
- `outputs/metrics.json`
- `outputs/metrics_per_label.csv`
- `outputs/fig_confusion_matrices.png` (2x3 grid of 2x2 confusion matrices)
- `outputs/fig_per_label_prf.png`
- `outputs/fig_loss_curves.png`
- `outputs/report_comparison.md`

---

## 6. Running the Web Application

Start the local server with Uvicorn:
```bash
uvicorn app.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to:
- **Interactive Web UI:** `http://127.0.0.1:8000/`
- **Interactive Swagger API Documentation:** `http://127.0.0.1:8000/docs`

---

## 7. API Usage Examples

### `GET /health`
```bash
curl -X GET http://127.0.0.1:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "labels": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
  "model_version": "Model V4"
}
```

### `POST /predict`
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "People should respect each others beliefs and treat each other kindly."}'
```

### `POST /predict-batch`
```bash
curl -X POST http://127.0.0.1:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Have a great day!", "You are an idiot!"]}'
```

### `POST /predict-cyberbully` (Dataset 1 Secondary Model)
```bash
curl -X POST http://127.0.0.1:8000/predict-cyberbully \
  -H "Content-Type: application/json" \
  -d '{"text": "I hate you because of your religion and race!"}'
```
```json
{
  "text": "I hate you because of your religion and race!",
  "cyberbully_detected": true,
  "top_class": "religion",
  "confidence": 0.8124,
  "confidence_percent": 81.24,
  "probabilities": {
    "not_cyberbullying": 0.0821,
    "ethnicity/race": 0.0912,
    "gender/sexual": 0.0143,
    "religion": 0.8124
  }
}
```

---

## 8. Limitations & Responsible AI

1. **Extreme Class Imbalance:** Certain categories like `threat` represent <0.5% of the data. While Model V4's `pos_weight` significantly elevates recall, rare-label precision remains sensitive to vocabulary noise.
2. **Context and Sarcasm:** Isolated sentence analysis cannot observe multi-turn conversational history, irony, reclaimed in-group phrasing, or cultural nuance.
3. **Fixed 0.5 Threshold:** A universal threshold is deployed for simplicity and standard benchmark comparison; however, per-label validation-tuned thresholds (see `outputs/metrics_tuned_thresholds.json`) can be configured for specific operational risk profiles.
4. **Advisory Decision Support:** Model outputs are strictly assistive recommendations and should never act as autonomous censors without human oversight.
