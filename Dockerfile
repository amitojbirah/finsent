# ============ Stage 1: builder — install dependencies ============
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Everything gets installed into one self-contained virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
# 1) CPU-only PyTorch first (much smaller than the CUDA build)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Core dependencies only (cached unless pyproject.toml changes)
WORKDIR /build
COPY pyproject.toml ./
RUN mkdir -p src/finsent && touch src/finsent/__init__.py \
    && pip install . \
    && pip uninstall -y finsent

# 3) Our own code (fast layer, rebuilt on every code change)
COPY src ./src
RUN pip install --no-deps .


# ============ Stage 2: runtime — what actually ships ============
FROM python:3.10-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    HF_HUB_OFFLINE=1

# Don't run as root inside the container
RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY artifacts/serving_model /app/model

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "finsent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]