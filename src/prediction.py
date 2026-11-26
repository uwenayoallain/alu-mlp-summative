"""Single-image prediction with a thread-safe, hot-swappable model."""

from __future__ import annotations

import threading
import time

import numpy as np

from src.config import CLASS_LABELS, CLASS_NAMES, MODEL_PATH
from src.preprocessing import load_image


class Predictor:
    def __init__(self, path=MODEL_PATH):
        self._lock = threading.Lock()
        # One inference at a time per process: concurrent requests queue instead of each
        # allocating TensorFlow buffers (which exhausts memory under load). Scale out with containers.
        self._infer = threading.Lock()
        self.path = path
        self.model = None
        self.loaded_at: float | None = None
        self.reload()

    def reload(self):
        from src.model import load_model

        model = load_model(self.path)
        model(np.zeros((1, *model.input_shape[1:]), np.float32))  # warm up so the first request is fast
        with self._lock:
            self.model = model
            self.loaded_at = time.time()

    def swap(self, model):
        """Install an already-built model (used right after retraining)."""
        with self._lock:
            self.model = model
            self.loaded_at = time.time()

    def predict_array(self, x: np.ndarray) -> np.ndarray:
        """Predict one image at a time: same input shape as warm-up, so memory stays flat."""
        with self._lock:
            model = self.model
        with self._infer:
            return np.concatenate([model(x[i:i + 1].astype(np.float32), training=False).numpy() for i in range(len(x))])

    def predict_bytes(self, data: bytes) -> dict:
        probs = self.predict_array(load_image(data)[None])[0]
        idx = int(probs.argmax())
        return {
            "class": CLASS_NAMES[idx],
            "label": CLASS_LABELS[CLASS_NAMES[idx]],
            "confidence": round(float(probs[idx]), 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
        }


def predict_file(path) -> dict:
    """Convenience for scripts and the notebook: python -m src.prediction <image>."""
    with open(path, "rb") as f:
        return Predictor().predict_bytes(f.read())


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(predict_file(sys.argv[1]), indent=2))
