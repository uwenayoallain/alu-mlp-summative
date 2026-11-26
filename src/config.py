"""Shared paths and settings for the coffee leaf pipeline."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

MODEL_PATH = MODELS_DIR / "coffee_leaf_model.keras"
MODEL_H5_PATH = MODELS_DIR / "coffee_leaf_model.h5"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
# Frozen-backbone embeddings of a replay sample and of the test set. Retraining
# reuses them so it only has to run the backbone on newly uploaded images.
REPLAY_EMB_PATH = MODELS_DIR / "replay_embeddings.npz"
TEST_EMB_PATH = MODELS_DIR / "test_embeddings.npz"
DATASET_STATS_PATH = MODELS_DIR / "dataset_stats.json"

CLASS_NAMES = ["cercospora", "healthy", "miner", "phoma", "rust"]
CLASS_LABELS = {
    "cercospora": "Cercospora leaf spot",
    "healthy": "Healthy",
    "miner": "Leaf miner",
    "phoma": "Brown leaf spot (Phoma)",
    "rust": "Coffee leaf rust",
}
IMG_SIZE = 224
SEED = 42

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'app.db'}")
# Unused uploaded images needed before retraining starts on its own.
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "40"))
IMAGE_LIMIT_MB = int(os.getenv("IMAGE_LIMIT_MB", "15"))
