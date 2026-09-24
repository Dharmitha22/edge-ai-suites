"""
Copyright (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

Linux replacement for the Windows `monitoring.monitor` module.

The Windows implementation spawns in-process collector threads
(monitoring/scripts/{common,windows}/collect_*.py) and serves their CSVs
per-session. On Linux there is no equivalent GPU/NPU/power collector to
spawn in-process, so this module instead proxies to the `metrics-collector`
sidecar (see docker-compose.yml: `image: intel/retail-benchmark:...`, no
custom build) which already runs those collectors continuously via
supervisord AND serves them over HTTP itself (performance-tools' own
live-metrics/metrics_api.py) - qmassa / Intel NPU tool / Intel PCM / sar /
free, the same tools `performance-tools` uses for offline benchmarking in
loss-prevention.

Behavioral differences from the Windows module, by design:
  * Collection is global and continuous (started with the container), not
    per-session. `start_monitoring`/`stop_monitoring` are therefore no-ops
    here, kept only so `api/endpoints.py` doesn't need an `if platform`
    branch. `metrics_logs_dir` is accepted (for call-signature parity) but
    unused.
  * `get_metrics` returns a live window of the sidecar's in-memory/on-disk
    series, not a from-the-start-of-session replay.

If the sidecar is unreachable (not started, still booting, wrong URL), every
function fails soft and returns the same empty-series shape the frontend
already treats as "no data yet" (see ui/src/services/api.ts:getResourceMetrics).
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

METRICS_COLLECTOR_URL = os.environ.get(
    "METRICS_COLLECTOR_URL", "http://metrics-collector:9000"
).rstrip("/")
_TIMEOUT_SECONDS = float(os.environ.get("METRICS_COLLECTOR_TIMEOUT", "3"))

_EMPTY_METRICS = {
    "cpu_utilization": [],
    "gpu_utilization": [],
    "npu_utilization": [],
    "memory": [],
    "power": [],
}


def start_monitoring(metrics_logs_dir: str = "./logs") -> None:
    """No-op: the metrics-collector sidecar's collectors run continuously."""
    logger.info(
        "start_monitoring(%s): metrics-collector sidecar collects continuously; "
        "nothing to start locally.",
        metrics_logs_dir,
    )


def stop_monitoring() -> None:
    """No-op: see start_monitoring."""
    logger.info("stop_monitoring(): no local collectors to stop.")


def is_monitoring_active() -> bool:
    """Best-effort liveness check against the sidecar's /health endpoint."""
    try:
        resp = httpx.get(f"{METRICS_COLLECTOR_URL}/health", timeout=_TIMEOUT_SECONDS)
        return resp.status_code == 200
    except httpx.HTTPError as e:
        logger.debug("metrics-collector health check failed: %s", e)
        return False


def get_metrics(metrics_logs_dir: str = "./logs") -> dict:
    """Fetch the current time-series window from the metrics-collector sidecar.

    `metrics_logs_dir` is accepted for call-signature parity with the Windows
    module (api/endpoints.py passes SessionPaths.utilization_logs_dir(...))
    but is not used: the sidecar is not session-scoped.
    """
    try:
        resp = httpx.get(f"{METRICS_COLLECTOR_URL}/metrics", timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        logger.warning(
            "metrics-collector unreachable at %s: %s", METRICS_COLLECTOR_URL, e
        )
        return dict(_EMPTY_METRICS)


def get_platform_info() -> dict:
    """Fetch the hardware summary (Processor/iGPU/NPU/Memory/Storage) from the
    metrics-collector sidecar. Returns {} if the sidecar is unreachable, so
    callers can fall back to local detection (see utils/platform_info.py).
    """
    try:
        resp = httpx.get(f"{METRICS_COLLECTOR_URL}/platform-info", timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPError as e:
        logger.warning(
            "metrics-collector platform-info unreachable at %s: %s",
            METRICS_COLLECTOR_URL, e,
        )
        return {}
