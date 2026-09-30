# syntax=docker/dockerfile:1.7
# One image for api, worker and migrations. Models are NOT baked in: they are downloaded
# once into the `model_cache` volume (HF_HOME=/models) and reused across restarts.
FROM python:3.12-slim AS base

ARG TORCH_VARIANT=cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH HF_HOME=/models TOKENIZERS_PARALLELISM=false

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first (cached layer). The lockfile pins CPU-only torch wheels.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project && \
    if [ "$TORCH_VARIANT" != "cpu" ]; then \
      uv pip install --python /opt/venv/bin/python --reinstall torch \
        --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"; \
    fi

COPY alembic.ini ./
COPY app ./app
COPY evaluation ./evaluation

RUN useradd --system --uid 10001 --home /app appuser && mkdir -p /models && chown -R appuser /models /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
