
import os
import logging
from typing import List

# --- Load .env.local if present (before any config/env usage) ---
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env.local", override=False)
except ImportError:
    pass  # If python-dotenv not installed, skip (should be in deps)

# --- Set HuggingFace model cache dir if EMBED_MODEL_CACHE is set ---
_model_cache = os.getenv("EMBED_MODEL_CACHE")
if _model_cache:
    os.environ["HF_HOME"] = _model_cache
    os.environ["TRANSFORMERS_CACHE"] = _model_cache

try:
    import torch  # optional at import time
except Exception:  # pragma: no cover
    torch = None

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# ----------------------------------------------------------------------------
# App & Config
# ----------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("embedding-server")


APP_TITLE = os.getenv("EMBED_APP_TITLE", "Embedding API")
APP_DESC = os.getenv(
    "EMBED_APP_DESC",
    "API to generate sentence embeddings using Sentence-Transformers."
)
APP_VERSION = os.getenv("EMBED_APP_VERSION", "1.1.0")

MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3-large")
# User requested CPU due to VRAM; default to CPU. Allow override via env.
CFG_DEVICE = os.getenv("EMBED_DEVICE", "cpu").lower()
MAX_ITEMS = int(os.getenv("EMBED_MAX_ITEMS", "128"))
MAX_TEXT_LEN = int(os.getenv("EMBED_MAX_TEXT_LEN", "2048"))  # per text, characters
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
SKIP_MODEL_LOAD = os.getenv("EMBED_SKIP_MODEL_LOAD", "0") in {"1", "true", "yes"}



app = FastAPI(title=APP_TITLE, description=APP_DESC, version=APP_VERSION)

# CORS: allow all by default; tighten in production via env if needed
allow_origins = os.getenv("EMBED_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allow_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def resolve_device() -> str:
    if CFG_DEVICE == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available; falling back to CPU.")
    return "cpu"


@app.on_event("startup")
def load_model_on_startup() -> None:
    device = resolve_device()
    app.state.device = device
    if SKIP_MODEL_LOAD:
        logger.info("EMBED_SKIP_MODEL_LOAD=1 detected; skipping real model load (using dummy model).")

        class _DummyModel:
            def __init__(self, dim: int = 1024):
                self.dim = dim

            def encode(self, texts: List[str], normalize_embeddings: bool = True, batch_size: int = 32, show_progress_bar: bool = False):
                import numpy as np
                rs = np.random.RandomState(42)
                out = rs.randn(len(texts), self.dim).astype("float32")
                if normalize_embeddings:
                    norms = np.linalg.norm(out, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    out = out / norms
                return out

        app.state.model = _DummyModel()
        app.state.model_name = f"dummy::{MODEL_NAME}"
        logger.info("Dummy model ready.")
        return

    logger.info("Loading model '%s' on device=%s ...", MODEL_NAME, device)
    try:
        # Import here to avoid requiring torch during dummy runs
        from sentence_transformers import SentenceTransformer
            hf_token = os.getenv("HF_TOKEN")
            model = SentenceTransformer(
                MODEL_NAME,
                device=device,
                use_auth_token=hf_token if hf_token else None
            )
        app.state.model = model
        app.state.model_name = MODEL_NAME
        logger.info("Model loaded and ready.")
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        # Keep a flag to indicate readiness failure
        app.state.model = None
        app.state.model_name = MODEL_NAME


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": getattr(app.state, "device", None),
        "model": getattr(app.state, "model_name", None),
        "ready": getattr(app.state, "model", None) is not None,
    }


@app.get("/ready")
def ready() -> dict:
    return {
        "ready": getattr(app.state, "model", None) is not None,
        "device": getattr(app.state, "device", None),
    }


class EmbeddingRequest(BaseModel):
    texts: List[str]


def _validate_texts(texts: List[str]) -> None:
    if not texts:
        raise HTTPException(status_code=400, detail="'texts' must be a non-empty list of strings.")
    if len(texts) > MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Too many items: {len(texts)} > {MAX_ITEMS}.")
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise HTTPException(status_code=400, detail=f"Item at index {i} is not a string.")
        if len(t) == 0:
            raise HTTPException(status_code=400, detail=f"Text at index {i} is empty.")
        if len(t) > MAX_TEXT_LEN:
            raise HTTPException(status_code=400, detail=f"Text at index {i} exceeds {MAX_TEXT_LEN} characters.")


@app.post("/embed/")
async def create_embeddings(request: EmbeddingRequest) -> dict:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready.")

    _validate_texts(request.texts)

    try:
        embeddings = await run_in_threadpool(
            model.encode,
            request.texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.exception("Embedding failed: %s", exc)
        raise HTTPException(status_code=500, detail="Embedding failed.")

    embeddings_list = embeddings.tolist()
    return {
        "object": "list",
        "model": getattr(app.state, "model_name", MODEL_NAME),
        "device": getattr(app.state, "device", None),
        "data": [
            {"object": "embedding", "embedding": emb, "index": i}
            for i, emb in enumerate(embeddings_list)
        ],
    }