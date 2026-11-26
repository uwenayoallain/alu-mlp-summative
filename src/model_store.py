"""Keep model versions in the database so the served model survives restarts.

Only the classifier head changes between versions (retraining keeps the MobileNetV2 backbone frozen),
so a version is its head weights plus its metadata. On startup the API restores the newest version.
"""

from __future__ import annotations

import io
import json

import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.config import METADATA_PATH
from src.database import ModelVersion, session


def _pack(head) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, *head.get_weights())
    return buf.getvalue()


def _unpack(blob: bytes) -> list[np.ndarray]:
    d = np.load(io.BytesIO(blob))
    return [d[f"arr_{i}"] for i in range(len(d.files))]


def record_version(head, meta: dict, source: str):
    """Store a version. Ignores a duplicate, e.g. two replicas seeding v1 at the same time."""
    try:
        with session() as s:
            s.add(ModelVersion(version=meta["version"], source=source, metadata_json=json.dumps(meta),
                               head_weights=_pack(head)))
            s.commit()
    except IntegrityError:
        pass


def restore_latest(predictor) -> int | None:
    """Seed the database with the bundled model if empty, else load the newest stored head."""
    from src.model import read_metadata, split_model

    _, head = split_model(predictor.model)
    with session() as s:
        latest = s.scalars(select(ModelVersion).order_by(ModelVersion.version.desc()).limit(1)).first()
    file_meta = read_metadata()
    if latest is None:
        record_version(head, file_meta, file_meta.get("trained_on", "notebook"))
        return file_meta.get("version")
    if latest.version != file_meta.get("version"):
        head.set_weights(_unpack(latest.head_weights))
        METADATA_PATH.write_text(json.dumps(json.loads(latest.metadata_json), indent=2))
    return latest.version
