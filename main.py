import os
import logging
import logging.config
from typing import List, Union, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import base64
import numpy as np
import tiktoken

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
    # Use HF_HOME (preferred). Avoid setting TRANSFORMERS_CACHE to prevent
    # deprecation warnings from transformers (TRANSFORMERS_CACHE is deprecated).

# Read logging overrides from environment early so they can be applied
LOG_FILE = os.getenv("LOG_FILE")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


try:
    import torch  # optional at import time
except Exception:  # pragma: no cover
    torch = None

# Configure logging
try:
    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger('embedding-server')
except Exception:
    # Fallback: respect LOG_FILE/LOG_LEVEL env if provided
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    handlers = [logging.StreamHandler()]
    if LOG_FILE:
        handlers.append(logging.FileHandler(LOG_FILE))
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s", handlers=handlers)
    logger = logging.getLogger("embedding-server")

# ----------------------------------------------------------------------------
# App & Config
# ----------------------------------------------------------------------------


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
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
    else:
        logger.info("Loading model '%s' on device=%s ...", MODEL_NAME, device)
        try:
            # Import here to avoid requiring torch during dummy runs
            from sentence_transformers import SentenceTransformer
            # Read huggingface hub token from the standard env var.
            # Only `HUGGINGFACE_HUB_TOKEN` is supported; HF_TOKEN has been removed.
            hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
            if hf_token:
                # nothing to change; ensure availability for huggingface internals
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

            model = SentenceTransformer(
                MODEL_NAME,
                device=device,
            )
            app.state.model = model
            app.state.model_name = MODEL_NAME
            logger.info("Model loaded and ready.")
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)
            # Keep a flag to indicate readiness failure
            app.state.model = None
            app.state.model_name = MODEL_NAME

    yield

    # Cleanup (if needed)
    pass


app = FastAPI(title=APP_TITLE, description=APP_DESC, version=APP_VERSION, lifespan=lifespan)

# CORS: allow all by default; tighten in production via env if needed
# In production, specify exact origins instead of using "*"
allow_origins = os.getenv("EMBED_CORS_ORIGINS", "*")
allow_origins_list = [o.strip() for o in allow_origins.split(",")] if allow_origins != "*" else ["*"]

# Validate CORS origins in production
if allow_origins != "*" and len(allow_origins_list) == 0:
    logger.warning("EMBED_CORS_ORIGINS is set but contains no valid origins. Defaulting to '*' for development.")
    allow_origins_list = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
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


class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"


@app.post("/v1/embeddings")
async def openai_embeddings(request: OpenAIEmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Accepts: {
      "input": string | list[string],
      "model": string (optional),
      "encoding_format": "float" | "base64" (optional)
    }
    Returns: {
      "object": "list",
      "data": [...],
      "model": string,
      "usage": {"prompt_tokens": int, "total_tokens": int}
    }
    """
    model_obj = getattr(app.state, "model", None)
    model_name = getattr(app.state, "model_name", MODEL_NAME)
    if model_obj is None:
        raise HTTPException(status_code=503, detail="Model not ready. Please check the application logs for model loading errors.")

    # Validate model name if provided
    if request.model and request.model != model_name:
        return {
            "error": {
                "message": f"The model '{request.model}' does not match the loaded model '{model_name}'",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found"
            }
        }

    # Determine input list
    if isinstance(request.input, str):
        texts = [request.input]
    elif isinstance(request.input, list) and all(isinstance(t, str) for t in request.input):
        texts = request.input
    else:
        raise HTTPException(status_code=400, detail="'input' must be a string or list of strings.")

    # Validate texts (reuse existing logic)
    if not texts:
        raise HTTPException(status_code=400, detail="'input' must be a non-empty string or list of strings.")
    if len(texts) > MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Too many items: {len(texts)} > {MAX_ITEMS}.")
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise HTTPException(status_code=400, detail=f"Item at index {i} is not a string.")
        if len(t) == 0:
            raise HTTPException(status_code=400, detail=f"Text at index {i} is empty.")
        if len(t) > MAX_TEXT_LEN:
            raise HTTPException(status_code=400, detail=f"Text at index {i} exceeds {MAX_TEXT_LEN} characters.")

    try:
        embeddings = await run_in_threadpool(
            model_obj.encode,
            texts,
            normalize_embeddings=True,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.exception("Embedding failed: %s", exc)
        error_detail = f"Embedding failed: {str(exc)}"
        raise HTTPException(status_code=500, detail=error_detail)

    embeddings_list = embeddings.tolist()
    encoding_format = (request.encoding_format or "float").lower()
    data = []
    for i, emb in enumerate(embeddings_list):
        if encoding_format == "base64":
            arr = np.array(emb, dtype=np.float32)
            emb_bytes = arr.tobytes()
            emb_out = base64.b64encode(emb_bytes).decode("utf-8")
        else:
            emb_out = emb
        data.append({"object": "embedding", "embedding": emb_out, "index": i})

    # Calculate usage (token count) if possible
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = sum(len(enc.encode(t)) for t in texts)
    except Exception:
        prompt_tokens = 0
    usage = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}

    return {
        "object": "list",
        "data": data,
        "model": request.model or model_name,
        "usage": usage,
    }