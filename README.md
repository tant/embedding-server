Embedding Server (FastAPI + Sentence-Transformers)

Overview
- A simple, production-leaning API to generate sentence embeddings.
- Defaults to CPU to avoid GPU VRAM issues; you can opt-in to CUDA via env.

Features
- Async FastAPI endpoint with threadpool offloading.
- Model load on startup, health and readiness probes.
- Input validation (max items, max text length), CORS enabled.
- Configurable via environment variables.
- Enhanced error handling with detailed error messages.
- Improved logging with file output capability.
- Comprehensive test suite with coverage for error conditions.
- Graceful handling of model loading failures.

Requirements
- Python 3.11+
- PyTorch (CPU or CUDA build) and sentence-transformers.

Install with uv (recommended)
- Project already lists dependencies in `pyproject.toml`. Use uv to resolve and install into `.venv`.

```bash
# create/refresh .venv and install deps from pyproject
uv sync

# Install development dependencies (optional)
uv sync --extra dev

# activate venv (optional, but nice for shell tools)
source .venv/bin/activate
```



Quick run với uv (không cần kích hoạt venv thủ công)
---------------------------------------------------
```bash
# Port mặc định của uvicorn là 8000. Để đổi port, bạn PHẢI export PORT hoặc chỉ định --port khi chạy uvicorn.
# EMBED_PORT chỉ dùng cho app nội bộ, không ảnh hưởng đến uvicorn.

# Cách 1: export PORT từ .env.local (khuyên dùng)
export $(grep -v '^#' .env.local | xargs)
uv run uvicorn main:app --host 0.0.0.0 --workers 1

# Cách 2: chỉ định --port trực tiếp
uv run uvicorn main:app --host 0.0.0.0 --port 7979 --workers 1

# Nếu cần CUDA (đảm bảo torch build đúng)
EMBED_DEVICE=cuda uv run uvicorn main:app --host 0.0.0.0 --port 7979 --workers 1
```


Environment variables
---------------------
Bạn nên cấu hình các biến môi trường trong file `.env.local` (đã có sẵn mẫu, tự động bị ignore bởi git):

```env
# .env.local
EMBED_PORT=7979
EMBED_DEVICE=cpu
EMBED_MODEL_NAME=BAAI/bge-m3-unsupervised
EMBED_MAX_ITEMS=128
EMBED_MAX_TEXT_LEN=2048
EMBED_BATCH_SIZE=32
EMBED_CORS_ORIGINS=*
EMBED_SKIP_MODEL_LOAD=0
EMBED_MODEL_CACHE= # Optional: set model cache dir (HuggingFace cache)
```

Khi chạy bằng uv hoặc uvicorn, các biến trong `.env.local` sẽ tự động được nạp nếu bạn dùng [uv](https://github.com/astral-sh/uv) hoặc [python-dotenv]. Nếu không, bạn có thể export thủ công:

```bash
export $(grep -v '^#' .env.local | xargs)
```

Port mặc định của uvicorn là **8000**. Để đổi port, bạn phải export PORT hoặc chỉ định --port khi chạy uvicorn.
**Lưu ý:** EMBED_PORT chỉ dùng cho app nội bộ, không ảnh hưởng đến uvicorn.


Logging
-------
The application uses a logging configuration file (`logging.conf`) to manage log output. By default, logs are written to both the console and a file named `embedding-server.log` in the root directory. You can modify the `logging.conf` file to change the logging behavior, such as log file location, log levels, or formatting.

Run tests (optional)
--------------------
```bash
# Run tests
uv run pytest -v

# Run tests with coverage
uv run pytest -v --cov=main tests/

# Run tests with coverage and generate HTML report
uv run pytest -v --cov=main --cov-report=html tests/
```

Endpoints
- GET /health: Basic health info (model/device/ready).
- GET /ready: Readiness flag.
- POST /embed/: Body {"texts": ["..."]} returns embeddings.

Example request
```bash
curl -s http://localhost:8000/embed/ \
	-H 'Content-Type: application/json' \
	-d '{"texts": ["Xin chao", "Hello world!"]}' | jq
```

Notes
- When using GPU, keep workers=1 to avoid multiple processes loading the same model.
- For scaling, prefer multiple instances behind a load balancer rather than multi-process sharing a single GPU.
- The application gracefully handles model loading failures, logging the error and marking the service as not ready.
- CORS is configured to allow all origins by default, but you should specify exact origins in production for security.