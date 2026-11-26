"""FastAPI service: prediction, bulk upload, retraining, evaluation and monitoring.

Run locally:  uv run uvicorn src.api:app --port 8000
Docs:         http://localhost:8000/docs
"""

from __future__ import annotations

import io
import json
import os
import random
import socket
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Keep TensorFlow from grabbing every core of a shared host before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(int(os.getenv("TF_THREADS", "2")))
tf.config.threading.set_inter_op_parallelism_threads(1)

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from sqlalchemy import desc, func, select  # noqa: E402

from src import retrain  # noqa: E402
from src.config import (CLASS_LABELS, CLASS_NAMES, DATASET_STATS_PATH, IMAGE_LIMIT_MB, RETRAIN_THRESHOLD,  # noqa: E402
                        STATIC_DIR, TEST_DIR, TEST_EMB_PATH)
from src.database import (PredictionLog, TrainingRun, Upload, engine, fail_interrupted_runs, init_db,  # noqa: E402
                          pending_upload_count, session)
from src.model import evaluate_predictions, read_metadata, split_model  # noqa: E402
from src.prediction import Predictor  # noqa: E402
from src.preprocessing import IMAGE_EXTS, is_valid_image, list_images, load_image  # noqa: E402

STARTED_AT = time.time()
init_db()
fail_interrupted_runs()
predictor = Predictor()
from src.model_store import restore_latest  # noqa: E402

restore_latest(predictor)
app = FastAPI(title="Coffee Leaf Disease API",
              description="Predict coffee leaf diseases from photos, upload new labelled data and retrain the model.",
              version="1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_stale_ui(request, call_next):
    """Make browsers revalidate the dashboard files so a redeploy shows up immediately."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["monitoring"])
def health():
    """Liveness and model up-time."""
    meta = read_metadata()
    return {
        "status": "ok",
        "instance": socket.gethostname(),
        "started_at": _iso(STARTED_AT),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "model_version": meta.get("version"),
        "model_loaded_at": _iso(predictor.loaded_at) if predictor.loaded_at else None,
        "retraining": retrain.is_running(),
        "database": engine.dialect.name,
    }


@app.get("/model", tags=["monitoring"])
def model_info():
    """Metadata of the model being served: version, test metrics and version history."""
    meta = read_metadata()
    meta["class_labels"] = CLASS_LABELS
    return meta


@app.post("/predict", tags=["prediction"])
def predict(file: UploadFile = File(..., description="Photo of one coffee leaf")):
    data = file.file.read()
    if not data or not is_valid_image(data):
        raise HTTPException(400, "Not a readable image file")
    t = time.perf_counter()
    result = predictor.predict_bytes(data)
    latency = (time.perf_counter() - t) * 1000
    with session() as s:
        s.add(PredictionLog(predicted=result["class"], confidence=result["confidence"], latency_ms=latency,
                            model_version=read_metadata().get("version")))
        s.commit()
    return {**result, "latency_ms": round(latency, 1), "model_version": read_metadata().get("version"),
            "instance": socket.gethostname()}


def _iter_upload_images(files: list[UploadFile], label: str | None):
    """Yield (filename, label, bytes). A zip may hold one folder per class name."""
    for f in files:
        data = f.file.read()
        if (f.filename or "").lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    p = Path(name)
                    if p.suffix.lower() not in IMAGE_EXTS or name.startswith("__MACOSX"):
                        continue
                    folder = p.parent.name.lower()
                    cls = folder if folder in CLASS_NAMES else label
                    yield p.name, cls, z.read(name)
        else:
            yield f.filename or "image", label, data


@app.post("/upload", tags=["retraining"])
def upload(files: list[UploadFile] = File(..., description="Images, or zip files with one folder per class"),
           label: str | None = Form(None, description=f"Class of the images: one of {CLASS_NAMES}. "
                                                       "Not needed for a zip with class folders.")):
    """Bulk upload of labelled images. Each image is validated and saved to the database for retraining."""
    if label is not None and label not in CLASS_NAMES:
        raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    saved, rejected = {c: 0 for c in CLASS_NAMES}, []
    with session() as s:
        for name, cls, data in _iter_upload_images(files, label):
            if cls is None:
                rejected.append({"file": name, "reason": "no label (choose a class or use class folders)"})
            elif len(data) > IMAGE_LIMIT_MB * 1024 * 1024:
                rejected.append({"file": name, "reason": f"larger than {IMAGE_LIMIT_MB} MB"})
            elif not is_valid_image(data):
                rejected.append({"file": name, "reason": "not a readable image"})
            else:
                s.add(Upload(filename=name, label=cls, image=data, size_bytes=len(data)))
                saved[cls] += 1
        s.commit()
    pending = pending_upload_count()
    auto = False
    if pending >= RETRAIN_THRESHOLD and not retrain.is_running():
        auto = retrain.start_in_background("auto", predictor)
    return {"saved": saved, "saved_total": sum(saved.values()), "rejected": rejected, "pending_for_retraining": pending,
            "retrain_threshold": RETRAIN_THRESHOLD, "auto_retrain_started": auto}


@app.get("/uploads", tags=["retraining"])
def uploads_summary():
    with session() as s:
        rows = s.execute(select(Upload.label, Upload.used_in_run.is_(None), func.count())
                         .group_by(Upload.label, Upload.used_in_run.is_(None))).all()
        recent = s.execute(select(Upload.id, Upload.filename, Upload.label, Upload.size_bytes, Upload.uploaded_at,
                                  Upload.used_in_run).order_by(desc(Upload.id)).limit(20)).all()
    pending = {c: 0 for c in CLASS_NAMES}
    used = {c: 0 for c in CLASS_NAMES}
    for lbl, is_pending, n in rows:
        (pending if is_pending else used)[lbl] = n
    return {"pending": pending, "used": used, "pending_total": sum(pending.values()),
            "retrain_threshold": RETRAIN_THRESHOLD,
            "recent": [dict(r._mapping) for r in recent]}


@app.post("/retrain", tags=["retraining"])
def trigger_retrain():
    """Start retraining on the uploaded images (runs in the background)."""
    if retrain.is_running():
        raise HTTPException(409, "A retraining job is already running")
    if pending_upload_count() == 0:
        raise HTTPException(400, "Upload some labelled images first")
    retrain.start_in_background("manual", predictor)
    return {"started": True}


@app.get("/retrain/status", tags=["retraining"])
def retrain_status(limit: int = 10):
    with session() as s:
        runs = s.scalars(select(TrainingRun).order_by(desc(TrainingRun.id)).limit(limit)).all()
    return {"running": retrain.is_running(),
            "runs": [{c.name: getattr(r, c.name) for c in TrainingRun.__table__.columns} for r in runs]}


@app.get("/evaluate", tags=["monitoring"])
def evaluate(n_images: int = Query(30, ge=0, le=300)):
    """Evaluate the model being served now.

    * full test set: the served head on the stored backbone features of all test images
    * live check: n_images random test photos sent through the full prediction pipeline
    """
    t = time.perf_counter()
    model = predictor.model
    _, head = split_model(model)
    d = __import__("numpy").load(TEST_EMB_PATH)
    full = evaluate_predictions(d["y"].astype(int), head(d["x"].astype("float32"), training=False).numpy())
    items = random.sample(list_images(TEST_DIR), min(n_images, len(list_images(TEST_DIR))))
    live = None
    if items:
        import numpy as np

        x = np.stack([load_image(p) for p, _ in items]).astype("float32")
        y = np.array([lbl for _, lbl in items])
        live = evaluate_predictions(y, predictor.predict_array(x))
    return {"model_version": read_metadata().get("version"), "test_set": full, "live_check": live,
            "seconds": round(time.perf_counter() - t, 2)}


@app.get("/stats/dataset", tags=["insights"])
def dataset_stats():
    return json.loads(DATASET_STATS_PATH.read_text())


@app.get("/stats/predictions", tags=["monitoring"])
def prediction_stats():
    with session() as s:
        total = s.scalar(select(func.count(PredictionLog.id))) or 0
        by_class = dict(s.execute(select(PredictionLog.predicted, func.count()).group_by(PredictionLog.predicted)).all())
        avg = s.scalar(select(func.avg(PredictionLog.latency_ms)))
        recent = s.execute(select(PredictionLog.created_at, PredictionLog.predicted, PredictionLog.confidence,
                                  PredictionLog.latency_ms).order_by(desc(PredictionLog.id)).limit(50)).all()
    return {"total": total, "by_class": {c: by_class.get(c, 0) for c in CLASS_NAMES},
            "avg_latency_ms": round(avg, 1) if avg else None,
            "recent": [dict(r._mapping) for r in recent]}
