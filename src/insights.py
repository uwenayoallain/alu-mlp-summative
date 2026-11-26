"""Per-image colour and texture features used for the dataset visualisations.

Every BRACOL photo is a single leaf on a white sheet, so the leaf is separated
from the background with a simple rule (background = bright and unsaturated).
Features measured on the leaf pixels only:

* leaf_area     - share of the frame covered by the leaf
* green_ratio   - share of leaf pixels whose hue is green (healthy tissue)
* lesion_ratio  - share of leaf pixels whose hue is yellow/orange/brown (symptoms)
* brightness    - mean HSV value of the leaf
* texture       - mean gradient magnitude on the leaf (spots and trails add edges)
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from src.config import CLASS_NAMES, DATASET_STATS_PATH, TEST_DIR, TRAIN_DIR
from src.preprocessing import list_images, load_image

FEATURES = ["leaf_area", "green_ratio", "lesion_ratio", "brightness", "texture"]


def image_features(rgb: np.ndarray) -> dict:
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    leaf = (s > 0.18) | (v < 0.55)
    n = max(int(leaf.sum()), 1)
    green = leaf & (h >= 65) & (h <= 170)
    lesion = leaf & ((h < 55) | (h > 330)) & (s > 0.25)
    grey = rgb.astype(np.float32).mean(-1)
    gy, gx = np.gradient(grey)
    return {
        "leaf_area": float(leaf.mean()),
        "green_ratio": float(green.sum() / n),
        "lesion_ratio": float(lesion.sum() / n),
        "brightness": float(v[leaf].mean()) if leaf.any() else 0.0,
        "texture": float(np.hypot(gx, gy)[leaf].mean()) if leaf.any() else 0.0,
    }


def compute_feature_table():
    import pandas as pd

    rows = []
    for split, folder in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        for path, y in list_images(folder):
            rows.append({"split": split, "class": CLASS_NAMES[y], "file": path.name,
                         **image_features(load_image(path))})
    return pd.DataFrame(rows)


def build_dataset_stats(df=None) -> dict:
    """Aggregate features per class for the web dashboard charts."""
    df = compute_feature_table() if df is None else df
    stats = {
        "counts": {s: {c: int(((df.split == s) & (df["class"] == c)).sum()) for c in CLASS_NAMES}
                   for s in ["train", "test"]},
        "features": {},
    }
    for f in FEATURES:
        per_class = {}
        for c in CLASS_NAMES:
            vals = df.loc[df["class"] == c, f].to_numpy()
            q = np.percentile(vals, [10, 25, 50, 75, 90])
            per_class[c] = {"mean": float(vals.mean()), "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3],
                            "p90": q[4]}
        stats["features"][f] = per_class
    # 20-bin histograms of lesion share, per class, for the dashboard
    bins = np.linspace(0, 0.6, 21)
    stats["lesion_hist"] = {"bins": bins.tolist(),
                            **{c: np.histogram(df.loc[df["class"] == c, "lesion_ratio"].clip(0, 0.6), bins)[0].tolist()
                               for c in CLASS_NAMES}}
    DATASET_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_STATS_PATH.write_text(json.dumps(stats, indent=2, default=float))
    return stats


if __name__ == "__main__":
    table = compute_feature_table()
    print(table.groupby("class")[FEATURES].mean().round(3))
    build_dataset_stats(table)
