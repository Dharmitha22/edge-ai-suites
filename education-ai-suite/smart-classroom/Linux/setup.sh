#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Smart Classroom Linux setup entry point (task 0.A.3 / 0.B.2).
# Replaces setup-smart-classroom.ps1. For now it provisions the Node.js toolchain
# the UI build needs; venv bootstrap / pip install / config prompts are layered on
# in 0.A.3.
#
# Usage:
#   ./setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Node.js toolchain for the UI build (0.B.2) ------------------------------
log "Checking Node.js toolchain for the UI build…"
NODE_SETUP="${SCRIPT_DIR}/scripts/node-setup.sh"
[[ -f "$NODE_SETUP" ]] || die "Missing ${NODE_SETUP}"
bash "$NODE_SETUP"

log "✅ setup complete."
