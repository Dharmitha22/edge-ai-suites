#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Provision the OVMS model repository (task 0.C.1):
#   export text_gen / embeddings / rerank to OpenVINO IR and assemble
#   models/ovms/ + config.json with the three servables.
#
# Thin wrapper around `python -m utils.ovms_provision`. Reused by
# setup-ovms-baremetal.sh (0.C.1a) and runnable standalone.
#
# Usage:
#   scripts/provision-ovms-models.sh                 # provision all three
#   scripts/provision-ovms-models.sh --force         # re-export even if IR exists
#   scripts/provision-ovms-models.sh --config-only   # only rewrite config.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$HERE/.." && pwd)"
cd "$APP_ROOT"

# Activate the app virtualenv if present (created by setup.sh, foundation 0.A.3).
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PY="${PYTHON:-python3}"
exec "$PY" -m utils.ovms_provision "$@"
