# Embedding Server (FastAPI + Sentence-Transformers)

A production-ready API server for generating sentence embeddings using FastAPI and Sentence-Transformers.

Repository: https://github.com/tant/embedding-server

## Overview

This is a simple, production-leaning API to generate sentence embeddings. It defaults to CPU to avoid GPU VRAM issues; you can opt-in to CUDA via environment variables.

## Features

- **Async FastAPI endpoint** with threadpool offloading
- **Model load on startup** with health and readiness probes
- **Input validation** (max items, max text length) with CORS enabled
- **Configurable** via environment variables
- **Enhanced error handling** with detailed error messages
- **Improved logging** with file output capability
- **Comprehensive test suite** with coverage for error conditions
- **Graceful handling** of model loading failures

## Requirements

- Python 3.11+
- PyTorch (CPU or CUDA build) and sentence-transformers

## Installation with uv (recommended)

The project already lists dependencies in `pyproject.toml`. Use uv to resolve and install into `.venv`.

```bash
# Create/refresh .venv and install deps from pyproject
uv sync

# Install development dependencies (optional)
uv sync --extra dev

# Activate venv (optional, but nice for shell tools)
source .venv/bin/activate
```

## Quick Run with uv

You can run the server without manually activating the virtual environment:

```bash
# Default uvicorn port is 8000. To change the port, you MUST export PORT or specify --port when running uvicorn.
# EMBED_PORT is only used internally and does not affect uvicorn.

# Method 1: Export PORT from .env.local (recommended)
export $(grep -v '^#' .env.local | xargs)
# Prefer running uvicorn as a module when using `uv` so the server is executed
# from the uv-managed environment. Using the plain `uv run uvicorn ...` form
# can fail if the `uvicorn` script isn't available on PATH inside the run
# environment; the `-m` form avoids that by executing the installed package.
uv run -m uvicorn main:app --host 0.0.0.0 --workers 1

# Method 2: Specify --port directly
# Use the module form to ensure uv runs the package from its environment:
uv run -m uvicorn main:app --host 0.0.0.0 --port 7979 --workers 1

# For CUDA support (ensure torch build is correct)
EMBED_DEVICE=cuda uv run uvicorn main:app --host 0.0.0.0 --port 7979 --workers 1
```

## Environment Variables

You should configure environment variables in the `.env.local` file (a sample is already provided and automatically ignored by git):

```env
# .env.local
EMBED_DEVICE=cpu
EMBED_MODEL_NAME=BAAI/bge-m3-unsupervised
EMBED_MAX_ITEMS=128
EMBED_MAX_TEXT_LEN=2048
EMBED_BATCH_SIZE=32
EMBED_CORS_ORIGINS=*
EMBED_SKIP_MODEL_LOAD=0
EMBED_MODEL_CACHE= # Optional: path to a local HF cache/snapshot (see notes below)
EMBED_APP_TITLE=Embedding API
EMBED_APP_DESC=API to generate sentence embeddings using Sentence-Transformers.
EMBED_APP_VERSION=1.1.0
```

When running with uv or uvicorn, variables in `.env.local` will be automatically loaded if you use [uv](https://github.com/astral-sh/uv) or [python-dotenv]. Otherwise, you can export them manually:

```bash
export $(grep -v '^#' .env.local | xargs)
```

Note: The default uvicorn port is **8000**. To change the port, you must export PORT or specify --port when running uvicorn. 
**Important:** EMBED_PORT is only used internally and does not affect uvicorn.

HuggingFace cache & token notes
--------------------------------
- If you set `EMBED_MODEL_CACHE`, the application maps it to `HF_HOME` so HuggingFace tools use that directory. Avoid using `TRANSFORMERS_CACHE` directly as it is deprecated.
- To authenticate private models or increase rate limits, set a token in the environment. The app reads `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` and will export it as `HUGGINGFACE_HUB_TOKEN` before loading models. Example:

```env
HUGGINGFACE_HUB_TOKEN=hf_xxx...
```

`.env.local.sample` is provided at the project root with a full example and usage notes — copy it to `.env.local` and edit values for your deployment.

## Logging

The application uses a logging configuration file (`logging.conf`) to manage log output. By default, logs are written to both the console and a file named `embedding-server.log` in the root directory. You can modify the `logging.conf` file to change the logging behavior, such as log file location, log levels, or formatting.

## Running Tests (Optional)

```bash
# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest -v --cov=main tests/

# Run tests with coverage and generate HTML report
uv run pytest -v --cov=main --cov-report=html tests/
```

## API Endpoints

- `GET /health`: Basic health info (model/device/ready)
- `GET /ready`: Readiness flag
- `POST /embed/`: Body `{"texts": ["..."]}` returns embeddings

### Example Request

```bash
curl -s http://localhost:8000/embed/ \
    -H 'Content-Type: application/json' \
    -d '{"texts": ["Xin chao", "Hello world!"]}' | jq
```

## Notes

- When using GPU, keep `workers=1` to avoid multiple processes loading the same model
- For scaling, prefer multiple instances behind a load balancer rather than multi-process sharing a single GPU
- The application gracefully handles model loading failures, logging the error and marking the service as not ready
- CORS is configured to allow all origins by default, but you should specify exact origins in production for security