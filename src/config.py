import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
CB_DATASET_CSV = DATA_DIR / "cb_multi_labeled_balanced.csv"

OUTPUTS_DIR = BASE_DIR / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
MODEL_V4_DIR = MODELS_DIR / "jigsaw_v4"
CYBERBULLY_MODEL_DIR = MODELS_DIR / "cyberbully_v1"

# Reproducibility
SEED = 42

# Jigsaw 6 Targets
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]
NUM_LABELS = len(LABEL_COLUMNS)

# Model & Tokenizer
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"

# Hyperparameters (V4 standard)
MAX_LEN = 128
BATCH_SIZE = 16  # Safe for 6GB RTX 3050 laptop GPU
GRAD_ACCUM_STEPS = 2  # Effective batch size = 32
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
DEFAULT_THRESHOLD = 0.5

# Pos_weight max clip
POS_WEIGHT_MAX_CLIP = 10.0

# Moderation Thresholds & Rules
CRITICAL_LABELS = {"severe_toxic", "threat", "identity_hate"}
SCORE_HIGH_THRESHOLD = 0.80
SCORE_REVIEW_MIN = 0.30
SCORE_REVIEW_MAX = 0.50

SEVERITY_LEVELS = ["None", "Low", "Moderate", "High", "Critical"]
MODERATION_ACTIONS = ["allow", "review", "warn", "hide", "escalate"]
