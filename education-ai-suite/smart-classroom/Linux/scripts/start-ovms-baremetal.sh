#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Bring up one OVMS instance (baremetal) on :9000 (task 0.C.1 / runtime R4–R5).
# Serves the three OpenAI-compatible endpoints (/v3/chat/completions,
# /v3/embeddings, /v3/rerank) + GET /v2/health/ready from models/ovms/config.json.
#
# Thin launcher: exports the env the native binary needs, then execs `ovms`
# (so systemd / the caller owns the process). Reused by ovms.service and start.sh.
#
# Port/config resolution (first match wins):
#   OVMS_PORT env  →  serving.base_url port in config.yaml  →  9000
#   OVMS_CONFIG env →  models/ovms/config.json
#
# Usage:
#   scripts/start-ovms-baremetal.sh                # foreground (exec)
#   OVMS_PORT=9000 scripts/start-ovms-baremetal.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$HERE/.." && pwd)"
cd "$APP_ROOT"

OVMS_PREFIX="${OVMS_PREFIX:-/opt/ovms}"
OVMS_BIN="${OVMS_BIN:-${OVMS_PREFIX}/bin/ovms}"

log()  { printf '\033[1;34m[ovms]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[ovms]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -x "$OVMS_BIN" ]] || die "OVMS binary not found at ${OVMS_BIN}. Run scripts/setup-ovms-baremetal.sh first."

# --- resolve REST port from config.yaml serving.base_url (best effort) -------
resolve_port() {
  [[ -n "${OVMS_PORT:-}" ]] && { echo "$OVMS_PORT"; return; }
  local port
  port="$(python3 - <<'PY' 2>/dev/null || true
from urllib.parse import urlparse
try:
    from utils.config_loader import config
    url = getattr(getattr(config, "serving", None), "base_url", "") or ""
    p = urlparse(url).port
    print(p or "")
except Exception:
    print("")
PY
)"
  echo "${port:-9000}"
}

REST_PORT="$(resolve_port)"
CONFIG_PATH="${OVMS_CONFIG:-${APP_ROOT}/models/ovms/config.json}"

[[ -f "$CONFIG_PATH" ]] || die "OVMS config not found at ${CONFIG_PATH}. Run scripts/provision-ovms-models.sh first."

# --- env the python_on native binary needs before exec ----------------------
export LD_LIBRARY_PATH="${OVMS_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${OVMS_PREFIX}/bin:${PATH}"
export PYTHONPATH="${OVMS_PREFIX}/lib/python:${PYTHONPATH:-}"

log "Starting OVMS on :${REST_PORT}  (config: ${CONFIG_PATH})"
log "  endpoints: /v3/chat/completions · /v3/embeddings · /v3/rerank · /v2/health/ready"

exec "$OVMS_BIN" \
  --rest_port "$REST_PORT" \
  --rest_bind_address "${OVMS_BIND:-127.0.0.1}" \
  --config_path "$CONFIG_PATH"
