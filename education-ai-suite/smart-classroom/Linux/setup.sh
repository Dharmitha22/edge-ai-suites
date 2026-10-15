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

# --- Git submodules (vendored audio-analyzer + performance-tools) ------------
# Audio Analyzer is pinned by the parent gitlink; only its service is checked out.
if git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  log "Initializing git submodules (edge-ai-libraries, performance-tools)…"
  git -C "$SCRIPT_DIR" submodule sync -- edge-ai-libraries
  git -C "$SCRIPT_DIR" submodule update --init edge-ai-libraries
  git -C "$SCRIPT_DIR/edge-ai-libraries" sparse-checkout init --no-cone
  git -C "$SCRIPT_DIR/edge-ai-libraries" sparse-checkout set --no-cone \
    '/*' \
    '!/*' \
    '/microservices/' \
    '!/microservices/*/' \
    '/microservices/audio-analyzer/'
  git -C "$SCRIPT_DIR" submodule update --init performance-tools
else
  log "Not a git checkout — skipping submodule init."
fi

# --- Node.js toolchain for the UI build (0.B.2) ------------------------------
log "Checking Node.js toolchain for the UI build…"
NODE_SETUP="${SCRIPT_DIR}/scripts/node-setup.sh"
[[ -f "$NODE_SETUP" ]] || die "Missing ${NODE_SETUP}"
bash "$NODE_SETUP"

log "✅ setup complete."
