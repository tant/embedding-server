Embedding Server (FastAPI + Sentence-Transformers)

Overview
- A simple, production-leaning API to generate sentence embeddings.
- Defaults to CPU to avoid GPU VRAM issues; you can opt-in to CUDA via env.

Features
- Async FastAPI endpoint with threadpool offloading.
- Model load on startup, health and readiness probes.
- Input validation (max items, max text length), CORS enabled.
- Configurable via environment variables.

Requirements
- Python 3.11+
- PyTorch (CPU or CUDA build) and sentence-transformers.

Install with uv (recommended)
- Project already lists dependencies in `pyproject.toml`. Use uv to resolve and install into `.venv`.

```bash
# create/refresh .venv and install deps from pyproject
uv sync

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
+EMBED_DEVICE=cpu
+EMBED_MODEL_NAME=BAAI/bge-m3-large
+EMBED_MAX_ITEMS=128
+EMBED_MAX_TEXT_LEN=2048
+EMBED_BATCH_SIZE=32
+EMBED_CORS_ORIGINS=*
+EMBED_SKIP_MODEL_LOAD=0
Khi chạy bằng uv hoặc uvicorn, các biến trong `.env.local` sẽ tự động được nạp nếu bạn dùng [uv](https://github.com/astral-sh/uv) hoặc [python-dotenv]. Nếu không, bạn có thể export thủ công:

```bash
export $(grep -v '^#' .env.local | xargs)
```

Port mặc định của uvicorn là **8000**. Để đổi port, bạn phải export PORT hoặc chỉ định --port khi chạy uvicorn.
**Lưu ý:** EMBED_PORT chỉ dùng cho app nội bộ, không ảnh hưởng đến uvicorn.
```

Khi chạy bằng uv hoặc uvicorn, các biến trong `.env.local` sẽ tự động được nạp nếu bạn dùng [uv](https://github.com/astral-sh/uv) hoặc [python-dotenv]. Nếu không, bạn có thể export thủ công:

```bash
export $(grep -v '^#' .env.local | xargs)
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

Run tests (optional)
```bash
uv run pytest -q
```

Notes
- When using GPU, keep workers=1 to avoid multiple processes loading the same model.
- For scaling, prefer multiple instances behind a load balancer rather than multi-process sharing a single GPU.

