# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Lightweight UI-serving skeleton for the Smart Classroom container.
#
# Serves the built React SPA (``ui/dist``) plus ``/health``, ``/metrics``, and
# ``/platform-info`` (proxied straight to the metrics-collector sidecar via
# monitoring/monitor.py) WITHOUT importing the heavy model stack
# (torch/openvino/paddleocr/...). It exists for the containerization skeleton
# phase, where the OVMS, model-downloader, and content-search containers are
# not yet built, so the app image can boot, render the UI, and show the live
# Resource Utilization dashboard on its own.
#
# Once the full backend image is wired up, swap the container CMD from
# ``uvicorn skeleton_server:app`` to ``uvicorn main:app`` and drop this module
# from the runtime stage.

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse

from monitoring.monitor import get_metrics, get_platform_info

# Directory holding the Vite build. Defaults to ``<this dir>/ui/dist`` so it
# matches the layout baked into the image; override with ``UI_DIST_DIR``.
UI_DIST = Path(
    os.environ.get("UI_DIST_DIR", Path(__file__).resolve().parent / "ui" / "dist")
)

app = FastAPI(title="Smart Classroom (UI skeleton)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
async def health() -> JSONResponse:
    """Backend liveness probe the SPA pings at ``/health``."""
    return JSONResponse({"status": "ok", "mode": "ui-skeleton"})


@app.get("/metrics", include_in_schema=False)
async def metrics() -> JSONResponse:
    """Live Resource Utilization dashboard data, proxied from metrics-collector.

    Session-less here (unlike the full backend's /metrics): the sidecar's
    collectors run continuously regardless of session state, so there is no
    session-scoped variant to serve in the skeleton phase.
    """
    return JSONResponse(get_metrics())


@app.get("/platform-info", include_in_schema=False)
async def platform_info() -> JSONResponse:
    """Hardware summary, proxied from metrics-collector.

    Skips the full backend's asr_model/summarizer_model merge (utils/platform_info.py)
    since no models are loaded in the skeleton phase.
    """
    return JSONResponse(get_platform_info())


def _mount_spa() -> None:
    """Serve the built SPA with an index.html fallback for client-side routes."""
    index_file = UI_DIST / "index.html"
    if not index_file.is_file():
        # No build present (e.g. running the module outside the image): expose
        # only /health so the container still starts and stays diagnosable.
        return

    assets_dir = UI_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    dist_root = UI_DIST.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        # Serve a real build file when it exists (favicon, manifest, ...),
        # otherwise fall back to index.html for client-side routing / refresh.
        candidate = (dist_root / full_path).resolve()
        if candidate.is_file() and str(candidate).startswith(str(dist_root)):
            return FileResponse(str(candidate))
        return FileResponse(str(index_file))


_mount_spa()
