# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Video processing Telegram bot
#
#   docker build -t video-bot .
#   docker compose up -d --build        (recommended - also starts the local
#                                        Bot API server, see docker-compose.yml)
#
# The image contains FFmpeg and the pre-downloaded vocal-separation weights,
# so the first job does not have to download anything.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# ffmpeg/ffprobe are used by the whole pipeline - and Demucs itself shells out
# to ffmpeg for all audio decoding/encoding.  libgomp1 is required by the
# PyTorch CPU wheels (OpenMP).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- python dependencies ---------------------------------------------------
# TORCH_INDEX_URL selects the PyTorch build:
#   https://download.pytorch.org/whl/cpu     CPU only (~250 MB, default)
#   https://download.pytorch.org/whl/cu126   CUDA 12.6 (use docker-compose.gpu.yml)
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.1,<3" \
 && pip install -r requirements.txt

COPY . .

# --- vocal-separation model ------------------------------------------------
# Baked into the image so a fresh container can separate vocals immediately.
# scripts/docker-entrypoint.sh copies this into the ./models volume on the
# first start (the mounted volume would otherwise hide it).
RUN python scripts/download_model.py --target /opt/model_cache

RUN mkdir -p temp output logs models \
 && chmod +x scripts/docker-entrypoint.sh

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["python", "-m", "bot.telegram_bot"]
