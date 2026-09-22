#!/usr/bin/env bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# One-time OVMS baremetal install (task 0.C.1a / runtime R4).
# Installs the OVMS *python_on* package (VLM chat templates need Jinja; the
# python_off build drops system-message + tool support), its runtime libs, and
# the Python deps OVMS loads at request time. GPU/NPU drivers are checked, not
# installed (follow OpenVINO "Additional Configurations for Hardware").
#
# Idempotent: re-running skips an already-extracted /opt/ovms unless --force.
#
# Usage:
#   sudo scripts/setup-ovms-baremetal.sh                 # install to /opt/ovms
#   sudo OVMS_VERSION=2026.3 scripts/setup-ovms-baremetal.sh
#   sudo OVMS_PREFIX=/opt/ovms --force
set -euo pipefail

OVMS_VERSION="${OVMS_VERSION:-2026.3}"
OVMS_PREFIX="${OVMS_PREFIX:-/opt/ovms}"
JINJA_VERSION="${JINJA_VERSION:-3.1.6}"
MARKUPSAFE_VERSION="${MARKUPSAFE_VERSION:-3.0.2}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

log()  { printf '\033[1;34m[setup-ovms]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup-ovms]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup-ovms]\033[0m %s\n' "$*" >&2; exit 1; }

# --- 0. sanity ---------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  die "Run as root (sudo): installs to ${OVMS_PREFIX} and uses apt."
fi

# --- 1. distro → package suffix ---------------------------------------------
# OVMS ships per-distro tarballs (ubuntu22 / ubuntu24 / rhel9).
distro_tag() {
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID}:${VERSION_ID%%.*}" in
      ubuntu:22) echo "ubuntu22"; return ;;
      ubuntu:24) echo "ubuntu24"; return ;;
      rhel:9|centos:9|rocky:9|almalinux:9) echo "redhat9"; return ;;
    esac
  fi
  echo ""
}
DISTRO="${OVMS_DISTRO:-$(distro_tag)}"
[[ -n "$DISTRO" ]] || die "Unsupported/undetected distro. Set OVMS_DISTRO=ubuntu22|ubuntu24|redhat9."

# --- 2. system libraries -----------------------------------------------------
log "Installing system prerequisites (libxml2, curl, python3-pip)…"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y libxml2 curl python3-pip
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y libxml2 curl python3-pip
else
  warn "No apt/dnf found — install libxml2, curl, python3-pip manually."
fi

# --- 3. download + extract OVMS python_on package ----------------------------
PKG="ovms_${DISTRO}_${OVMS_VERSION}.0_python_on.tar.gz"
URL="${OVMS_PACKAGE_URL:-https://github.com/openvinotoolkit/model_server/releases/download/v${OVMS_VERSION}/${PKG}}"

if [[ -x "${OVMS_PREFIX}/bin/ovms" && $FORCE -eq 0 ]]; then
  log "⚡ OVMS already installed at ${OVMS_PREFIX} (use --force to reinstall)."
else
  case "$URL" in
    https://*) : ;;
    *) die "OVMS_PACKAGE_URL must be https (got: ${URL})." ;;
  esac
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  log "⬇️  Downloading ${PKG}"
  curl -fSL "$URL" -o "${TMP}/${PKG}"
  log "📦 Extracting to $(dirname "${OVMS_PREFIX}")"
  # The tarball unpacks an 'ovms/' dir; place it at OVMS_PREFIX's parent.
  mkdir -p "$(dirname "${OVMS_PREFIX}")"
  tar -xzf "${TMP}/${PKG}" -C "$(dirname "${OVMS_PREFIX}")"
  [[ -x "${OVMS_PREFIX}/bin/ovms" ]] || die "Extraction did not produce ${OVMS_PREFIX}/bin/ovms."
fi

# --- 4. Python deps OVMS loads at runtime (python_on build) ------------------
log "Installing Python runtime deps (Jinja2, MarkupSafe)…"
pip3 install --upgrade "Jinja2==${JINJA_VERSION}" "MarkupSafe==${MARKUPSAFE_VERSION}"

# --- 5. GPU/NPU driver check (advisory only) --------------------------------
log "Checking Intel GPU/NPU availability…"
if [[ -e /dev/dri/renderD128 ]]; then
  log "  iGPU render node present: /dev/dri/renderD128"
else
  warn "  No /dev/dri/renderD128 — install Intel GPU drivers for GPU serving."
fi
if [[ -e /dev/accel/accel0 ]]; then
  log "  NPU device present: /dev/accel/accel0"
else
  warn "  No /dev/accel/accel0 — install the Intel NPU driver if NPU serving is needed."
fi

# --- 6. version smoke test ---------------------------------------------------
if LD_LIBRARY_PATH="${OVMS_PREFIX}/lib" "${OVMS_PREFIX}/bin/ovms" --version >/dev/null 2>&1; then
  log "✅ OVMS ${OVMS_VERSION} installed at ${OVMS_PREFIX}"
else
  warn "ovms --version failed; check LD_LIBRARY_PATH / package integrity."
fi

log "Next: scripts/start-ovms-baremetal.sh  (or enable ovms.service)"
