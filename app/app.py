import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import ToxicityPredictor, get_predictor

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Global predictor holder
predictor: Optional[ToxicityPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = get_predictor()
        print(f"Model successfully initialized on device: {predictor.device} (Version: {predictor.model_version})")
    except Exception as e:
        print(f"Warning during model initialization: {e}")
        predictor = None
    yield

app = FastAPI(
    title="AI-Powered Cyberbullying & Toxicity Predictor API",
    description="Contextual multi-label toxicity detection and moderation advice powered by fine-tuned DistilBERT (Model V4).",
    version="4.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Request Models
class PredictRequest(BaseModel):
    text: str = Field(..., description="The comment text to analyze (1 to 2000 characters)")

    @field_validator("text")
    def validate_text(cls, v: str):
        if v is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Text cannot be null.")
        stripped = v.strip()
        if len(stripped) == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Comment text cannot be empty or contain only whitespace. Please enter a valid comment."
            )
        if len(stripped) > 2000:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Comment text exceeds maximum length of 2000 characters (received {len(stripped)} chars)."
            )
        return stripped

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of comment strings to analyze (max 50 items)")

    @field_validator("texts")
    def validate_batch(cls, v: List[str]):
        if not v:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Batch texts list cannot be empty."
            )
        if len(v) > 50:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Batch size exceeds maximum limit of 50 items (received {len(v)})."
            )
        cleaned_list = []
        for i, item in enumerate(v):
            if item is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Item at index {i} cannot be null."
                )
            stripped = item.strip()
            if len(stripped) == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Item at index {i} cannot be empty or only whitespace."
                )
            if len(stripped) > 2000:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Item at index {i} exceeds limit of 2000 characters ({len(stripped)} chars)."
                )
            cleaned_list.append(stripped)
        return cleaned_list

@app.get("/", include_in_schema=False)
async def serve_index():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {"message": "AI Toxicity Predictor API is running. Visit /docs for the Swagger interface."}

@app.get("/health", tags=["System"])
async def health_check():
    global predictor
    if predictor is None:
        try:
            predictor = get_predictor()
        except Exception as e:
            print(f"Health check model initialization warning: {e}")
    is_loaded = predictor is not None
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "device": str(predictor.device) if is_loaded else "none",
        "labels": predictor.labels if is_loaded else [],
        "model_version": predictor.model_version if is_loaded else "unknown",
        "privacy_pipeline": {
            "pii_redaction_active": True,
            "storage_persistence": False,
            "placeholders": ["[PHONE]", "[EMAIL]", "[PERSON]", "[ADDRESS]"]
        }
    }

@app.get("/privacy-policy", tags=["System"])
async def get_privacy_policy():
    return {
        "policy_title": "AegisAI Privacy-Aware Data Processing & Sanitization Policy",
        "data_retention": "Zero Persistence. Comments are processed transiently in volatile RAM and purged immediately.",
        "pii_masking": "Incoming texts undergo deterministic PII redaction before transformer tokenization.",
        "redacted_entities": {
            "phone_numbers": "Masked with [PHONE]",
            "email_addresses": "Masked with [EMAIL]",
            "personal_names": "Masked with [PERSON]",
            "addresses_and_ips": "Masked with [ADDRESS]"
        },
        "logging_guarantee": "No raw comment text, IP addresses, or PII entities are stored in databases or log files.",
        "regulatory_alignment": ["GDPR Data Minimization (Art. 5)", "India DPDP Act (2023)"]
    }

@app.post("/predict", tags=["Inference"])
async def predict_single(req: PredictRequest):
    global predictor
    if predictor is None:
        try:
            predictor = get_predictor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Toxicity model is currently loading or unavailable: {str(e)}"
            )
    try:
        result = predictor.predict(req.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

@app.post("/predict-batch", tags=["Inference"])
async def predict_batch(req: PredictBatchRequest):
    global predictor
    if predictor is None:
        try:
            predictor = get_predictor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Toxicity model is currently loading or unavailable: {str(e)}"
            )
    try:
        results = predictor.predict_batch(req.texts)
        return {
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}"
        )

@app.post("/predict-cyberbully", tags=["Secondary Model (Dataset 1)"])
async def predict_cyberbully_endpoint(req: PredictRequest):
    from src.predict_cyberbully import get_cyberbully_predictor
    cb_pred = get_cyberbully_predictor()
    if cb_pred is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Secondary cyberbullying model is not currently available or trained."
        )
    try:
        return cb_pred.predict(req.text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cyberbullying inference error: {str(e)}"
        )

