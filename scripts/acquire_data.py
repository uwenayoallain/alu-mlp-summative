"""Data acquisition: download BRACOL from Mendeley Data and build data/train, data/test, data/new_data.

BRACOL (Krohling, Esgario & Ventura, 2019, DOI 10.17632/yy2k5y8mxg.1) holds smartphone photos
of Arabica coffee leaves on a white background, labelled by an expert with the predominant
biotic stress. The archive Mendeley serves is truncated (its checksum matches Mendeley's own,
yet the zip has no central directory), so the standard zipfile module cannot open it. This
script walks the local file headers instead, which recovers every complete image.

Split (stratified, seed 42): 20% test, then 10% of the rest held back as data/new_data. The
model never sees data/new_data during the initial training: it is the "new field data" that
gets uploaded through the UI to demonstrate retraining.

Run:  uv run python -m scripts.acquire_data
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import struct
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, ROOT, SEED

DATASET_ID = "yy2k5y8mxg"
FILES_API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?folder_id=root&version=1"
RAW_DIR = ROOT / "raw_downloads"
ZIP_PATH = RAW_DIR / "bracol.zip"
# BRACOL "predominant_stress" codes. Code 5 is not described by the authors and is dropped.
STRESS_TO_CLASS = {"0": "healthy", "1": "miner", "2": "rust", "3": "phoma", "4": "cercospora"}
SAVE_SIZE = (448, 224)  # keep the 2:1 leaf aspect at a size that is small enough to commit


def download():
    RAW_DIR.mkdir(exist_ok=True)
    if ZIP_PATH.exists():
        print("already downloaded:", ZIP_PATH)
        return
    with urllib.request.urlopen(FILES_API) as r:
        url = json.load(r)[0]["content_details"]["download_url"]
    print("downloading", url)
    with urllib.request.urlopen(url) as r, open(ZIP_PATH, "wb") as f:
        shutil.copyfileobj(r, f)


def iter_zip_entries(path: Path):
    """Yield (name, bytes) from zip local headers; works without a central directory."""
    data = path.read_bytes()
    pos = 0
    while True:
        pos = data.find(b"PK\x03\x04", pos)
        if pos < 0 or pos + 30 > len(data):
            return
        _, _, flags, method, _, _, _, csize, _, nlen, xlen = struct.unpack("<IHHHHHIIIHH", data[pos:pos + 30])
        name = data[pos + 30:pos + 30 + nlen].decode("utf-8", "replace")
        start = pos + 30 + nlen + xlen
        payload = data[start:start + csize]
        pos = start + csize if csize else pos + 4
        if flags & 0x08 or len(payload) < csize or not csize:
            continue  # streamed or cut-off entry: skip it
        if method == 8:
            try:
                payload = zlib.decompress(payload, -15)
            except zlib.error:
                continue
        elif method != 0:
            continue
        yield name, payload


def build_splits():
    entries = dict(iter_zip_entries(ZIP_PATH))
    csv_name = next(n for n in entries if n.endswith("leaf/dataset.csv"))
    rows = list(csv.DictReader(io.StringIO(entries[csv_name].decode())))
    images = {Path(n).stem: b for n, b in entries.items() if "/leaf/images/" in n and n.lower().endswith(".jpg")}

    items, skipped = [], Counter()
    for r in rows:
        cls = STRESS_TO_CLASS.get(r["predominant_stress"])
        img = images.get(r["id"])
        if cls is None:
            skipped["undocumented label 5"] += 1
        elif img is None:
            skipped["missing from truncated archive"] += 1
        else:
            try:
                Image.open(io.BytesIO(img)).verify()
                items.append((r["id"], cls, img))
            except Exception:
                skipped["corrupt image"] += 1
    print(f"{len(rows)} labelled rows, {len(items)} usable images, skipped: {dict(skipped)}")

    labels = [c for _, c, _ in items]
    rest, test = train_test_split(items, test_size=0.20, stratify=labels, random_state=SEED)
    train, new = train_test_split(rest, test_size=0.10, stratify=[c for _, c, _ in rest], random_state=SEED)

    summary = {}
    for split, group in [("train", train), ("test", test), ("new_data", new)]:
        out = DATA_DIR / split
        shutil.rmtree(out, ignore_errors=True)
        for img_id, cls, blob in group:
            (out / cls).mkdir(parents=True, exist_ok=True)
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(blob))).convert("RGB")
            if img.height > img.width:
                img = img.rotate(90, expand=True)
            img.resize(SAVE_SIZE, Image.Resampling.LANCZOS).save(out / cls / f"bracol_{img_id}.jpg", quality=90)
        summary[split] = dict(sorted(Counter(c for _, c, _ in group).items()))
    (DATA_DIR / "split_summary.json").write_text(json.dumps({"skipped": dict(skipped), **summary}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    download()
    build_splits()
