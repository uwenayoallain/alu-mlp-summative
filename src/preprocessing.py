"""Image loading, cleaning and dataset building.

The same `load_image` function is used for training data, uploaded retraining
data and single images sent for prediction, so every path sees identical
preprocessing:

1. decode the file and apply its EXIF rotation (phone photos),
2. convert to RGB (drops alpha channels and greyscale modes),
3. turn portrait shots to landscape, because every BRACOL leaf is photographed
   lying horizontally (2:1 frame),
4. resize to IMG_SIZE x IMG_SIZE.

Pixel scaling to [-1, 1] happens inside the model (a Rescaling layer), so raw
0-255 arrays go in everywhere.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.config import CLASS_NAMES, IMG_SIZE, SEED

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_pil(source: str | Path | bytes) -> Image.Image:
    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if img.height > img.width:
        img = img.rotate(90, expand=True)
    return img


def load_image(source: str | Path | bytes, size: int = IMG_SIZE) -> np.ndarray:
    """Return a (size, size, 3) uint8 array."""
    img = to_pil(source)
    img.draft("RGB", (size * 2, size * 2))  # fast JPEG downscale on decode for big phone photos
    return np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


def is_valid_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False


def list_images(folder: Path) -> list[tuple[Path, int]]:
    """(path, class index) pairs for a folder laid out as folder/<class>/*.jpg."""
    items = []
    for idx, name in enumerate(CLASS_NAMES):
        class_dir = folder / name
        if class_dir.is_dir():
            items += [(p, idx) for p in sorted(class_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return items


def load_folder(folder: Path, limit_per_class: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a class-folder directory into (uint8 images, int labels)."""
    items = list_images(folder)
    if limit_per_class:
        rng = random.Random(SEED)
        by_class: dict[int, list] = {}
        for p, y in items:
            by_class.setdefault(y, []).append((p, y))
        items = [it for group in by_class.values() for it in rng.sample(group, min(limit_per_class, len(group)))]
    x = np.stack([load_image(p) for p, _ in items]) if items else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    y = np.array([lbl for _, lbl in items], dtype=np.int64)
    return x, y


def make_tf_dataset(x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = False):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).map(lambda a, b: (tf.cast(a, tf.float32), b)).prefetch(2)  # fixed buffer: autotune grows memory


def augmentation_layers():
    """Leaves are photographed at any orientation and light level, so flip, rotate, zoom and relight."""
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(0.15, fill_mode="reflect"),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
