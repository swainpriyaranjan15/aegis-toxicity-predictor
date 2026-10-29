import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODEL_V4_DIR,
    LABEL_COLUMNS,
    DEFAULT_THRESHOLD,
    MAX_LEN,
    CRITICAL_LABELS,
    SCORE_HIGH_THRESHOLD,
    SCORE_REVIEW_MIN,
    SCORE_REVIEW_MAX
)
from src.preprocess import clean_text

class ToxicityPredictor:
    def __init__(self, model_dir_or_id: Union[str, Path] = None, threshold: float = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fallback to env var MODEL_ID if local model dir is missing
        if model_dir_or_id is None:
            if MODEL_V4_DIR.exists() and (MODEL_V4_DIR / "model_meta.json").exists():
                model_dir_or_id = MODEL_V4_DIR
            else:
                model_dir_or_id = os.environ.get("MODEL_ID", str(MODEL_V4_DIR))

        self.model_path = str(model_dir_or_id)
        self.meta = {}
        meta_file = Path(self.model_path) / "model_meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        self.max_len = self.meta.get("max_len", MAX_LEN)
        self.threshold = threshold if threshold is not None else self.meta.get("default_threshold", DEFAULT_THRESHOLD)
        self.labels = self.meta.get("label_order", LABEL_COLUMNS)
        self.model_version = self.meta.get("model_version", "Model V4")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def determine_moderation(self, detected_labels: List[str], score: float, highest_label: str) -> tuple[str, str, str]:
        """
        Ordered Moderation Policy:
        1. any of severe_toxic, threat, identity_hate detected -> Critical, escalate
        2. score >= 0.80 or 2+ labels detected -> High, hide
        3. any label detected (score 0.50-0.80) -> Moderate, warn
        4. no label detected but score 0.30-0.50 -> Low, review
        5. otherwise -> None, allow
        """
        # Rule 1: Critical threats or extreme toxicity
        if any(lbl in CRITICAL_LABELS for lbl in detected_labels):
            severity = "Critical"
            action = "escalate"
            crit_matches = [lbl for lbl in detected_labels if lbl in CRITICAL_LABELS]
            reason = f"High-risk severe content detected ({', '.join(crit_matches)}); escalated for priority human review."
            return severity, action, reason

        # Rule 2: High confidence toxic or multiple violations
        if score >= SCORE_HIGH_THRESHOLD or len(detected_labels) >= 2:
            severity = "High"
            action = "hide"
            if len(detected_labels) >= 2:
                reason = f"Multiple toxicity indicators detected ({', '.join(detected_labels)}); hidden to protect community."
            else:
                reason = f"Strong toxicity detected ({highest_label} score {score*100:.1f}%); content hidden."
            return severity, action, reason

        # Rule 3: Single moderate toxicity label detected
        if len(detected_labels) >= 1:
            severity = "Moderate"
            action = "warn"
            reason = f"Moderate toxicity detected ({', '.join(detected_labels)}); warning flag applied."
            return severity, action, reason

        # Rule 4: Borderline / ambiguous confidence
        if SCORE_REVIEW_MIN <= score <= SCORE_REVIEW_MAX:
            severity = "Low"
            action = "review"
            reason = f"Borderline language detected ({highest_label} score {score*100:.1f}%); queued for context review."
            return severity, action, reason

        # Rule 5: Clean / safe content
        return "None", "allow", "No toxic patterns detected; comment allowed."

    def predict(self, text: str) -> Dict[str, Any]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        from src.privacy import get_sanitizer
        sanitizer = get_sanitizer()

        sanitized_texts = []
        privacy_metas = []
        for t in texts:
            san_text, p_meta = sanitizer.redact(t)
            sanitized_texts.append(san_text)
            privacy_metas.append(p_meta)

        cleaned_texts = [clean_text(st) for st in sanitized_texts]

        start_time = time.perf_counter()
        encoded = self.tokenizer(
            cleaned_texts,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

        elapsed_total_ms = (time.perf_counter() - start_time) * 1000.0
        per_item_ms = round(elapsed_total_ms / max(len(texts), 1), 2)

        results = []
        for i, raw_text in enumerate(texts):
            p_meta = privacy_metas[i]
            sanitized_text = sanitized_texts[i]
            prob_dict = {col: round(float(probs[i][j]), 4) for j, col in enumerate(self.labels)}
            detected = [col for col in self.labels if prob_dict[col] >= self.threshold]

            # Highest label and confidence
            highest_label = max(prob_dict, key=prob_dict.get)
            max_prob = prob_dict[highest_label]
            score_percent = round(max_prob * 100.0, 2)

            prediction_label = "Toxic" if len(detected) > 0 else "Non-toxic"
            severity, action, reason = self.determine_moderation(detected, max_prob, highest_label)

            snippet = sanitized_text[:90] + ("..." if len(sanitized_text) > 90 else "")
            results.append({
                "text": sanitized_text,
                "text_snippet": snippet,
                "raw_text": raw_text,
                "prediction": prediction_label,
                "score": max_prob,
                "score_percent": score_percent,
                "highest_label": highest_label,
                "confidence": max_prob,
                "probabilities": prob_dict,
                "detected_labels": detected,
                "severity": severity,
                "moderation_action": action,
                "reason": reason,
                "threshold": self.threshold,
                "inference_ms": per_item_ms,
                "privacy": {
                    "pii_detected": p_meta["pii_detected"],
                    "redacted_text": sanitized_text,
                    "redaction_counts": p_meta["counts"],
                    "total_redactions": p_meta["total_redactions"],
                    "types_detected": p_meta["types_detected"]
                }
            })
        return results

# Singleton instance helper
_predictor_instance = None

def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ToxicityPredictor()
    return _predictor_instance

if __name__ == "__main__":
    predictor = ToxicityPredictor()
    sample_inputs = [
        "People should respect each other's beliefs and treat each other kindly.",
        "You are an absolute idiot and a worthless loser, get lost!",
        "The word 'idiot' is considered offensive in formal debates, though people use it casually."
    ]

    print("\n--- Model Inference & Moderation Checkpoint ---")
    for sample in sample_inputs:
        res = predictor.predict(sample)
        print(f"\nText: {res['text']}")
        print(f"Prediction: {res['prediction']} (Score: {res['score_percent']}%, Highest: {res['highest_label']})")
        print(f"Detected: {res['detected_labels']}")
        print(f"Severity: {res['severity']} | Action: {res['moderation_action']}")
        print(f"Reason: {res['reason']}")
        print(f"Latency: {res['inference_ms']} ms")
