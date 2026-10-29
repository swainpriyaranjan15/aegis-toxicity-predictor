import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CYBERBULLY_MODEL_DIR
from src.train_cyberbully import clean_tweet, CB_CLASSES

class CyberbullyPredictor:
    def __init__(self, model_dir=None):
        import os
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_dir is None:
            if CYBERBULLY_MODEL_DIR.exists() and (CYBERBULLY_MODEL_DIR / "config.json").exists():
                model_dir = CYBERBULLY_MODEL_DIR
            else:
                model_dir = os.environ.get("CYBERBULLY_MODEL_ID", "jintannn/cyberbully-multilabel-v1")
        
        self.model_path = str(model_dir)
        self.classes = CB_CLASSES

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict[str, Any]:
        from src.privacy import get_sanitizer
        sanitizer = get_sanitizer()
        sanitized_text, p_meta = sanitizer.redact(text)

        cleaned = clean_tweet(sanitized_text)
        inputs = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=96,
            padding=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_idx = int(probs.argmax())
        top_class = self.classes[top_idx]
        top_prob = float(probs[top_idx])
        is_cyberbullying = (top_class != "not_cyberbullying")

        prob_dict = {cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(self.classes)}

        return {
            "text": sanitized_text,
            "raw_text": text,
            "cyberbully_detected": is_cyberbullying,
            "top_class": top_class,
            "confidence": round(top_prob, 4),
            "confidence_percent": round(top_prob * 100.0, 2),
            "probabilities": prob_dict,
            "privacy": {
                "pii_detected": p_meta["pii_detected"],
                "redacted_text": sanitized_text,
                "redaction_counts": p_meta["counts"],
                "total_redactions": p_meta["total_redactions"],
                "types_detected": p_meta["types_detected"]
            }
        }

_cb_instance: Optional[CyberbullyPredictor] = None

def get_cyberbully_predictor() -> Optional[CyberbullyPredictor]:
    global _cb_instance
    if _cb_instance is None:
        try:
            _cb_instance = CyberbullyPredictor()
        except Exception as e:
            print(f"CyberbullyPredictor unavailable: {e}")
            _cb_instance = None
    return _cb_instance
