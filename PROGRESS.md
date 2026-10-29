# Project Progress: AI-Powered Cyberbullying / Toxicity Predictor with Contextual Moderation

| Phase | Description | Status | Key Numbers / Artifacts | How to Resume |
|---|---|---|---|---|
| **Phase 0** | Setup, Environment & Dataset Verification | **DONE** | CUDA: True (NVIDIA GeForce RTX 3050 6GB Laptop GPU), Rows: 159,571, Columns: 8 verified. PyTorch 2.11 + transformers. | Check with `python -c "import torch; print(torch.cuda.is_available())"` |
| **Phase 1** | Data preparation & stratified splitting (`preprocess.py`) | **DONE** | Clean rows: 159,571. Splits: 127,656 train (80%), 15,957 val (10%), 15,958 test (10%). Zero labels: 89.83%, 2+ labels: 6.18%. Generated `outputs/label_stats.json` & `fig_label_frequency.png`. | Re-run `python src/preprocess.py` |
| **Phase 2** | Model & Training loop with V4 pos_weight (`train.py`) | **DONE** | Full 3-epoch run completed in 37.72 min on RTX 3050 (mixed precision FP16, dynamic padding). Best Epoch: 3 (`Val Loss: 0.1055`, `Val Macro-F1: 0.6638`). Checkpoint saved to `models/jigsaw_v4/`. | Re-run `python src/train.py` or resume with `--resume` |
| **Phase 3** | Evaluation & metrics (`evaluate.py`) | **DONE** | Evaluated on 15,958 held-out test comments. **Hamming Acc: 0.9794** (matches reported ~0.98), **Subset Acc: 0.9087**, **Toxic F1: 0.8099**, **Mean ROC-AUC: 0.9903**, **Macro-F1: 0.6510** (0.6877 tuned). Figures generated. | Re-run `python src/evaluate.py` |
| **Phase 4** | Inference & Moderation logic (`predict.py`) | **DONE** | 5-tier ordered moderation policy (`escalate`, `hide`, `warn`, `review`, `allow`). Verified on canonical benign, abusive, and ambiguous comments. | Test with `python src/predict.py` |
| **Phase 5** | FastAPI Backend (`app/app.py`) | **DONE** | `/`, `/health`, `/predict`, `/predict-batch`, Swagger `/docs`. Strict 1-2000 char length & whitespace validation (HTTP 422). | Run `uvicorn app.app:app --reload` |
| **Phase 6** | Frontend Interface (`app/static/index.html`) | **DONE** | Modern single-page responsive UI with 6 dynamic probability bars, 50% threshold marker, severity badges, live examples, and advisory disclaimer. | Open `http://localhost:8000` |
| **Phase 7** | Tests, Documentation & Deployment files | **DONE** | `pytest -q` **11/11 PASSED (100% green)**. Latency: **Mean 9.28 ms, p95 14.63 ms** across 50 requests (`outputs/latency.json`). `outputs/sample_api_results.json` populated from live API. `render.yaml` & `requirements-cpu.txt` created. | Run `pytest -q` |
| **Phase 8** | Dataset 1 Cyberbullying model (`train_cyberbully.py`) | **DONE** | Trained 4-class softmax model on `cb_multi_labeled_balanced.csv` (`models/cyberbully_v1/`). Test Accuracy: 0.7733, Macro-F1: 0.6581. Generated confusion matrix & metrics. Integrated `/predict-cyberbully` & UI card. | Run `python src/train_cyberbully.py` |

## Active Compute Configuration
- **Hardware Tier:** Tier 1 — NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.8, PyTorch with Autocast FP16 + Dynamic Padding).
- **Default Seed:** 42.
- **Total Training Time:** 37.72 minutes across 3 full epochs.
