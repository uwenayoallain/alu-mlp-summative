"""Model definition, saving/loading and evaluation helpers.

Architecture: MobileNetV2 backbone (ImageNet weights, transfer learning) with a
small classification head:

    input (160x160x3, 0-255) -> [augmentation, training only] -> Rescaling to [-1, 1]
    -> MobileNetV2 (no top) -> GlobalAveragePooling -> Dropout -> Dense(5, softmax, L2)

The backbone and the head are separate named sub-models ("backbone" and "head")
so retraining can run the backbone once per new image and then fit only the head.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np

from src.config import CLASS_NAMES, IMG_SIZE, METADATA_PATH, MODEL_H5_PATH, MODEL_PATH

L2 = 1e-4
DROPOUT = 0.3


def build_backbone(trainable: bool = False):
    import tensorflow as tf

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), name="pixels")
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="to_minus1_1")(inputs)
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
                                             weights="imagenet")
    base.trainable = trainable
    x = base(x, training=False)  # keep BatchNorm in inference mode even when fine-tuning
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    return tf.keras.Model(inputs, x, name="backbone")


def build_head(n_features: int = 1280, n_classes: int = len(CLASS_NAMES)):
    import tensorflow as tf

    inputs = tf.keras.Input((n_features,), name="features")
    x = tf.keras.layers.Dropout(DROPOUT)(inputs)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax",
                                    kernel_regularizer=tf.keras.regularizers.l2(L2), name="probs")(x)
    return tf.keras.Model(inputs, outputs, name="head")


def build_model(augment: bool = True):
    import tensorflow as tf

    from src.preprocessing import augmentation_layers

    backbone, head = build_backbone(), build_head()
    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = augmentation_layers()(inputs) if augment else inputs
    outputs = head(backbone(x))
    return tf.keras.Model(inputs, outputs, name="coffee_leaf_classifier")


def split_model(model):
    """Return the (backbone, head) sub-models of a built or loaded classifier."""
    return model.get_layer("backbone"), model.get_layer("head")


def assemble(backbone, head):
    """Inference model without augmentation, built from trained parts."""
    import tensorflow as tf

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    return tf.keras.Model(inputs, head(backbone(inputs)), name="coffee_leaf_classifier")


def load_model(path=MODEL_PATH):
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def save_model(model, metrics: dict, extra: dict | None = None, h5: bool = True) -> dict:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    if h5:
        model.save(MODEL_H5_PATH)
    meta = read_metadata()
    version = meta.get("version", 0) + 1
    meta = {
        "version": version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "metrics": metrics,
        "history": meta.get("history", []) + [{"version": version, "metrics": metrics, **(extra or {})}],
        **(extra or {}),
    }
    METADATA_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def read_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Accuracy, loss, macro precision/recall/F1, per-class report and confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, log_loss,
                                 precision_score, recall_score)

    y_pred = probs.argmax(1)
    labels = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "loss": round(float(log_loss(y_true, np.clip(probs, 1e-7, 1), labels=labels)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "n_samples": int(len(y_true)),
        "per_class": classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                           output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
