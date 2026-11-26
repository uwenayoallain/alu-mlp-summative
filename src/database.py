"""SQL storage for uploaded training images, retraining runs and prediction logs.

Uploaded images are stored in the database itself (as bytes) together with
their label, so the retraining data survives independently of the file system.
Uses SQLite by default; set DATABASE_URL to a PostgreSQL URL to use Postgres.
"""

from __future__ import annotations

import re
import ssl
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from datetime import datetime, timezone

from sqlalchemy import (Boolean, DateTime, Float, Integer, LargeBinary, String, Text, create_engine, func,
                        select)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src.config import DATA_DIR, DATABASE_URL


def _now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Upload(Base):
    __tablename__ = "uploads"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(32), index=True)
    image: Mapped[bytes] = mapped_column(LargeBinary)
    size_bytes: Mapped[int] = mapped_column(Integer)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    used_in_run: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trigger: Mapped[str] = mapped_column(String(16))  # "manual" or "auto"
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued/running/done/failed
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    n_new_images: Mapped[int] = mapped_column(Integer, default=0)
    accuracy_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    accuracy_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    log: Mapped[str] = mapped_column(Text, default="")


class PredictionLog(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    predicted: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ModelVersion(Base):
    """Every served model version. The classifier head weights (about 25 KB) are stored here so the
    latest retrained model survives restarts on hosts with an ephemeral disk; the backbone never changes."""
    __tablename__ = "model_versions"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    source: Mapped[str] = mapped_column(String(16))  # "notebook" or "retrain"
    metadata_json: Mapped[str] = mapped_column(Text)  # the model_metadata.json written for this version
    head_weights: Mapped[bytes] = mapped_column(LargeBinary)


DATA_DIR.mkdir(parents=True, exist_ok=True)
def _engine_args(url: str) -> tuple[str, dict]:
    """SQLAlchemy URL and connect args for SQLite or PostgreSQL.

    PostgreSQL uses pg8000, a pure-Python driver that encrypts with Python's ssl module. C drivers
    (psycopg, libpq) load a native OpenSSL that clashes with the TLS library bundled in TensorFlow
    and crash the process on SSL connections. Like libpq's sslmode=require, the connection is
    encrypted without verifying the certificate; sslmode=disable turns SSL off.
    """
    if url.startswith("sqlite"):
        return url, {"check_same_thread": False}
    parts = urlsplit(re.sub(r"^postgres(ql)?://", "postgresql+pg8000://", url))
    params = dict(parse_qsl(parts.query))
    sslmode = params.pop("sslmode", "require")
    args = {}
    if sslmode != "disable":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        args["ssl_context"] = ctx
    return urlunsplit(parts._replace(query=urlencode(params))), args


_url, _connect_args = _engine_args(DATABASE_URL)
engine = create_engine(_url, connect_args=_connect_args, pool_pre_ping=True)


def init_db():
    Base.metadata.create_all(engine)
    if _url.startswith("sqlite"):
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")


def fail_interrupted_runs():
    """Mark runs left 'running' by a crash or restart as failed; their uploads stay available."""
    from sqlalchemy import update

    with session() as s:
        s.execute(update(TrainingRun).where(TrainingRun.status.in_(["queued", "running"]))
                  .values(status="failed", finished_at=_now(),
                          log=TrainingRun.log + "\ninterrupted: the server restarted before this run finished"))
        s.commit()


def session() -> Session:
    return Session(engine, expire_on_commit=False)


def pending_upload_count() -> int:
    with session() as s:
        return s.scalar(select(func.count(Upload.id)).where(Upload.used_in_run.is_(None))) or 0
