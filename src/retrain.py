"""Retraining with newly uploaded data, using the current model as the pre-trained starting point.

Steps (each one is logged to the training_runs table):

1. Load every upload not used in a previous run from the database.
2. Preprocess them with the same `load_image` used for training and prediction.
3. Load the current saved model and split it into its frozen MobileNetV2 backbone
   and its classification head. The backbone turns the new images into 1280-d features.
4. Continue training the existing head (its current weights are the starting point,
   not a fresh initialisation) on the new features plus a replay set of the original
   training features, so it learns the new data without forgetting the old.
5. Evaluate before and after on the fixed test set. The new model is saved and
   served only if its test accuracy does not drop by more than one point.

The replay and test features come from models/*.npz, written at training time,
so retraining runs the backbone only on the new images. That keeps it fast and
within the memory of a small cloud instance.
"""

from __future__ import annotations

import gc
import io
import threading
import traceback
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select

from src.config import CLASS_NAMES, REPLAY_EMB_PATH, RETRAIN_EPOCHS, SEED, TEST_EMB_PATH
from src.database import TrainingRun, Upload, session
from src.preprocessing import load_image

_lock = threading.Lock()
MAX_ACCURACY_DROP = 0.01


def is_running() -> bool:
    return _lock.locked()


def embed(backbone, images: np.ndarray, batch: int = 1) -> np.ndarray:
    """Backbone features. Batch 1 matches the prediction path, so no extra CPU kernel buffers are allocated."""
    out = [backbone(images[i:i + batch].astype(np.float32), training=False).numpy()
           for i in range(0, len(images), batch)]
    return np.concatenate(out) if out else np.zeros((0, 1280), np.float32)


def save_embeddings(path, x, y):
    np.savez_compressed(path, x=x.astype(np.float16), y=y.astype(np.int8))


def load_embeddings(path):
    d = np.load(path)
    return d["x"].astype(np.float32), d["y"].astype(np.int64)


def class_weights(y: np.ndarray) -> dict:
    counts = np.bincount(y, minlength=len(CLASS_NAMES)).astype(float)
    counts[counts == 0] = 1
    w = len(y) / (len(CLASS_NAMES) * counts)
    return {i: float(v) for i, v in enumerate(w)}


def fine_tune_head(head, x, y, epochs=RETRAIN_EPOCHS, lr=3e-4):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(SEED)
    head.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = head.fit(x, y, epochs=epochs, batch_size=32, validation_split=0.15, shuffle=True, verbose=0,
                    class_weight=class_weights(y),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                                                restore_best_weights=True)])
    return hist.history


def run_retraining(trigger: str = "manual", predictor=None) -> int:
    """Run one retraining job synchronously and return its run id."""
    from src.model import assemble, build_head, evaluate_predictions, load_model, read_metadata, save_model, split_model

    if not _lock.acquire(blocking=False):
        raise RuntimeError("A retraining job is already running")
    with session() as s:
        run = TrainingRun(trigger=trigger, status="running")
        s.add(run)
        s.commit()
        run_id = run.id
    log: list[str] = []

    def note(msg):
        log.append(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}")
        with session() as s:
            s.get(TrainingRun, run_id).log = "\n".join(log)
            s.commit()

    try:
        # 1. uploaded data from the database
        with session() as s:
            uploads = s.scalars(select(Upload).where(Upload.used_in_run.is_(None))).all()
        if not uploads:
            raise ValueError("No new uploaded images to train on")
        note(f"loaded {len(uploads)} new images from the database")

        # 2. preprocessing (same function as training and prediction)
        images, labels = [], []
        for u in uploads:
            try:
                images.append(load_image(io.BytesIO(u.image).getvalue()))
                labels.append(CLASS_NAMES.index(u.label))
            except Exception as e:  # skip unreadable files, keep the rest
                note(f"skipped {u.filename}: {e}")
        x_new_img, y_new = np.stack(images), np.array(labels)
        note(f"preprocessed {len(x_new_img)} images to {x_new_img.shape[1:]}; per class "
             f"{ {c: int((y_new == i).sum()) for i, c in enumerate(CLASS_NAMES)} }")

        # 3. current model as the pre-trained starting point. The frozen backbone is shared with
        #    the served model (read-only), and the head is copied with its current weights, so
        #    retraining never holds a second full network in memory.
        backbone, served_head = split_model(predictor.model if predictor is not None else load_model())
        head = build_head()
        head.set_weights(served_head.get_weights())
        x_new = embed(backbone, x_new_img)
        del x_new_img
        x_rep, y_rep = load_embeddings(REPLAY_EMB_PATH)
        x_test, y_test = load_embeddings(TEST_EMB_PATH)
        before = evaluate_predictions(y_test, head(x_test, training=False).numpy())
        note(f"current model v{read_metadata().get('version')} test accuracy {before['accuracy']:.4f}")

        # 4. continue training the head on new + replay features
        x_all, y_all = np.concatenate([x_rep, x_new]), np.concatenate([y_rep, y_new])
        hist = fine_tune_head(head, x_all, y_all)
        note(f"fine-tuned head for {len(hist['loss'])} epochs on {len(x_all)} samples "
             f"(final val_accuracy {hist['val_accuracy'][-1]:.4f})")

        # 5. evaluate and promote
        after = evaluate_predictions(y_test, head(x_test, training=False).numpy())
        promoted = after["accuracy"] >= before["accuracy"] - MAX_ACCURACY_DROP
        note(f"test accuracy after retraining {after['accuracy']:.4f} (F1 {after['f1']:.4f})")
        version = None
        if promoted:
            new_model = assemble(backbone, head)
            meta = save_model(new_model, after, extra={"trained_on": "retrain", "run_id": run_id,
                                                       "n_new_images": int(len(y_new))}, h5=False)
            version = meta["version"]
            save_embeddings(REPLAY_EMB_PATH, x_all, y_all)
            if predictor is not None:
                predictor.swap(new_model)
            note(f"promoted as model v{version} and now serving it")
        else:
            note("not promoted: accuracy dropped by more than one point; keeping the current model")

        with session() as s:
            r = s.get(TrainingRun, run_id)
            r.status, r.finished_at = "done", datetime.now(timezone.utc)
            r.n_new_images, r.promoted, r.model_version = len(y_new), promoted, version
            r.accuracy_before, r.accuracy_after, r.f1_after = before["accuracy"], after["accuracy"], after["f1"]
            for u in s.scalars(select(Upload).where(Upload.id.in_([u.id for u in uploads]))):
                u.used_in_run = run_id
            s.commit()
        gc.collect()
    except Exception as e:
        note(f"failed: {e}")
        traceback.print_exc()
        with session() as s:
            r = s.get(TrainingRun, run_id)
            r.status, r.finished_at = "failed", datetime.now(timezone.utc)
            s.commit()
    finally:
        _lock.release()
    return run_id


def start_in_background(trigger: str, predictor=None) -> bool:
    if is_running():
        return False
    threading.Thread(target=run_retraining, args=(trigger, predictor), daemon=True).start()
    return True


if __name__ == "__main__":
    from src.database import init_db

    init_db()
    print("run id", run_retraining("manual"))
