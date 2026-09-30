# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Smart Classroom — app image (containerization skeleton phase).
#
# Multi-stage build:
#   Stage 1 (ui-builder): compile the React/Vite SPA to /ui/dist.
#   Stage 2 (runtime):    a slim FastAPI container that serves that SPA plus
#                         /health, /metrics, and the transcription endpoints
#                         (/upload-audio, /transcribe, /live-transcript, and the
#                         /v1/realtime WebSocket). Transcription is delegated to
#                         the audio-analyzer microservice, so NO heavy model
#                         stack (torch/openvino/paddle) is installed here.
#
# The remaining heavy features (summary / mindmap / video-analytics / OCR) still
# require the full model stack; when that is ready, switch the runtime stage to
# install requirements.txt, copy the whole app source, and run `uvicorn main:app`
# instead of `uvicorn skeleton_server:app`.
#
# Build context is the Linux/ app root:
#   docker build -f docker/app.Dockerfile -t smart-classroom-app:local .

# ---- Stage 1: UI builder -----------------------------------------------------
FROM node:22-bookworm-slim AS ui-builder

# Proxy build args (forwarded from the host so npm can reach the registry behind
# a corporate proxy). Empty by default on direct-internet hosts.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY=""

WORKDIR /ui

# Copy lockfiles first for layer caching.
COPY ui/package.json ui/package-lock.json ./
RUN HTTP_PROXY="$HTTP_PROXY" HTTPS_PROXY="$HTTPS_PROXY" NO_PROXY="$NO_PROXY" npm ci

# Empty => relative, same-origin API calls (recommended for the baked-in image).
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY ui/ ./
RUN npm run build

# ---- Stage 2: Python runtime (UI skeleton) -----------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Proxy build args (forwarded from the host so pip can reach PyPI behind a
# corporate proxy). Scoped to the RUN below so they are never baked into the
# final image env. Empty by default on direct-internet hosts.
ARG HTTP_PROXY=""
ARG HTTPS_PROXY=""
ARG NO_PROXY=""

# Lightweight deps for the UI-serving + transcription-proxy server. The audio
# analyzer does the actual ASR out-of-process, so no torch/openvino here.
# Versions pinned to match requirements.txt so the runtime matches the full app.
RUN HTTP_PROXY="$HTTP_PROXY" HTTPS_PROXY="$HTTPS_PROXY" NO_PROXY="$NO_PROXY" \
    pip install --no-cache-dir \
        "fastapi==0.121.3" \
        "uvicorn==0.38.0" \
        "httpx>=0.27,<1.0" \
        "requests>=2.32,<3" \
        "websockets==15.0.1" \
        "python-multipart>=0.0.27" \
        "PyYAML==6.0.2"

# Non-root runtime user.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

# Skeleton server + built SPA (served from /app/ui/dist).
COPY docker/skeleton_server.py ./skeleton_server.py
COPY monitoring/ ./monitoring/
COPY --from=ui-builder /ui/dist ./ui/dist

# Transcription path (lightweight): the client + speaker-mapping layer, the WS
# proxy, and the config the server reads at startup. Only the modules the
# skeleton server imports are needed at runtime; the rest stay out of the image.
COPY utils/ ./utils/
COPY components/__init__.py components/asr_remote.py ./components/
COPY api/__init__.py api/proxy.py ./api/
COPY config.yaml runtime_config.yaml ./

RUN chown -R app:app /app
USER app

EXPOSE 8000

# Liveness probe (no curl in slim images).
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "skeleton_server:app", "--host", "0.0.0.0", "--port", "8000"]
