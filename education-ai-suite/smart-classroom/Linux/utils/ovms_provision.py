# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stand up the OVMS model repository (task 0.C.1).

Exports the three served capabilities — ``text_gen`` (VLM/LLM), ``embeddings`` and
``rerank`` — to OpenVINO IR and assembles ``models/ovms/`` + ``config.json`` with the
three mediapipe servables, as specified in ``docs/ovms-runtime-serving.md`` (R1).

The heavy lifting is done by OVMS' own ``export_model.py`` helper (it produces the IR +
mediapipe ``graph.pbtxt`` per servable). This module locates that helper, drives it once
per servable into the shared repo, and writes the canonical ``config.json``.

Run standalone::

    python -m utils.ovms_provision                 # provision all three servables
    python -m utils.ovms_provision --force         # re-export even if IR exists
    python -m utils.ovms_provision --config-only   # only (re)write config.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils.cli_utils import run_cli
from utils.config_loader import config

logger = logging.getLogger(__name__)

# --- Repository layout ------------------------------------------------------
SC_ROOT = Path(__file__).resolve().parents[1]
OVMS_MODELS_DIR = SC_ROOT / "models" / "ovms"
OVMS_TOOLS_DIR = OVMS_MODELS_DIR / "tools"
OVMS_CONFIG_PATH = OVMS_MODELS_DIR / "config.json"

# --- OVMS export tooling ----------------------------------------------------
# Pinned to the release referenced in docs/ovms-runtime-serving.md. Override with a
# vendored/verified local copy via OVMS_EXPORT_SCRIPT for air-gapped installs, or the
# source tag/URL via OVMS_VERSION / OVMS_EXPORT_SCRIPT_URL.
OVMS_VERSION = os.environ.get("OVMS_VERSION", "2026.3")
_EXPORT_SCRIPT_URL = os.environ.get(
    "OVMS_EXPORT_SCRIPT_URL",
    f"https://raw.githubusercontent.com/openvinotoolkit/model_server/"
    f"v{OVMS_VERSION}/demos/common/export_models/export_model.py",
)

# Default rerank model (config has no rerank block yet; see 0.C.3 config schema).
_DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-large"


@dataclass
class Servable:
    """One OVMS servable to export into the model repository."""

    kind: str  # export_model.py task: text_generation | embeddings | rerank
    source_model: str  # HF/ModelScope repo id
    target_device: str  # CPU | GPU | NPU
    weight_format: Optional[str] = None  # int4 | int8 | None (no quantization)
    model_name: str = field(default="")  # served name; defaults to basename of source

    def __post_init__(self) -> None:
        if not self.model_name:
            self.model_name = self.source_model.split("/")[-1]

    @property
    def model_dir(self) -> Path:
        return OVMS_MODELS_DIR / self.model_name

    @property
    def graph_path(self) -> str:
        # Relative to the repo root; this is what config.json references.
        return f"{self.model_name}/graph.pbtxt"

    def is_ready(self) -> bool:
        """True when a usable IR + graph already exist for this servable."""
        graph = self.model_dir / "graph.pbtxt"
        if not graph.is_file():
            return False
        # export_model.py writes the IR under a numbered version dir (e.g. 1/).
        return any(self.model_dir.glob("*/openvino_model.xml")) or any(
            self.model_dir.glob("*/*.xml")
        )


def _hub() -> str:
    """Return the configured model hub ('huggingface' | 'modelscope')."""
    return str(getattr(config.models, "model_hub", "huggingface")).lower()


def _text_gen_spec() -> Servable:
    tg = config.models.text_gen
    source = getattr(tg, "ovms_model", None) or getattr(tg, "vlm_name")
    return Servable(
        kind="text_generation",
        source_model=str(source),
        target_device=str(getattr(tg, "device", "GPU")).upper(),
        weight_format=str(getattr(tg, "weight_format", "int4")),
    )


def _embeddings_spec() -> Servable:
    emb = getattr(config.models, "embedding", None)
    source = getattr(emb, "ovms_model", None) or getattr(emb, "name", None) \
        or "BAAI/bge-large-en-v1.5"
    return Servable(
        kind="embeddings",
        source_model=str(source),
        target_device=str(getattr(emb, "device", "CPU")).upper() if emb else "CPU",
    )


def _rerank_spec() -> Servable:
    rr = getattr(config.models, "rerank", None)
    source = getattr(rr, "ovms_model", None) or getattr(rr, "name", None) \
        or _DEFAULT_RERANK_MODEL
    return Servable(
        kind="rerank",
        source_model=str(source),
        target_device=str(getattr(rr, "device", "CPU")).upper() if rr else "CPU",
    )


def build_specs() -> List[Servable]:
    """Assemble the three servable specs from config (with sane fallbacks)."""
    return [_text_gen_spec(), _embeddings_spec(), _rerank_spec()]


def ensure_export_script() -> Path:
    """Locate OVMS' ``export_model.py``, downloading a pinned copy if needed.

    Resolution order:
      1. ``OVMS_EXPORT_SCRIPT`` env var pointing at a local (vendored) copy.
      2. A previously cached copy under ``models/ovms/tools/``.
      3. Download from the pinned OVMS release tag.
    """
    override = os.environ.get("OVMS_EXPORT_SCRIPT")
    if override:
        p = Path(override).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"OVMS_EXPORT_SCRIPT points at a missing file: {p}"
            )
        return p

    cached = OVMS_TOOLS_DIR / "export_model.py"
    if cached.is_file():
        return cached

    OVMS_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "⬇️  Fetching OVMS export_model.py (pinned v%s)\n     from %s",
        OVMS_VERSION,
        _EXPORT_SCRIPT_URL,
    )
    try:
        # Pinned tag URL (https-only); for air-gapped installs set OVMS_EXPORT_SCRIPT
        # to a locally verified copy instead.
        if not _EXPORT_SCRIPT_URL.lower().startswith("https://"):
            raise ValueError("export script URL must be https")
        with urllib.request.urlopen(_EXPORT_SCRIPT_URL) as resp:  # noqa: S310 (pinned https)
            data = resp.read()
        cached.write_bytes(data)
    except Exception as e:
        raise RuntimeError(
            f"Could not obtain OVMS export_model.py from {_EXPORT_SCRIPT_URL}: {e}\n"
            "Set OVMS_EXPORT_SCRIPT to a local copy, or OVMS_EXPORT_SCRIPT_URL / "
            "OVMS_VERSION to a reachable source."
        ) from e
    logger.info("✅ Cached export helper at %s", cached)
    return cached


def export_servable(script: Path, spec: Servable, force: bool = False) -> None:
    """Export one servable's IR + graph into the OVMS repo (idempotent)."""
    if not force and spec.is_ready():
        logger.info("⚡ Servable '%s' already present — skipping export", spec.model_name)
        return

    OVMS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        spec.kind,
        "--source_model",
        spec.source_model,
        "--model_repository_path",
        str(OVMS_MODELS_DIR),
        "--model_name",
        spec.model_name,
        "--target_device",
        spec.target_device,
    ]
    if spec.weight_format:
        cmd += ["--weight-format", spec.weight_format]

    logger.info(
        "🚀 Exporting %s servable '%s' (%s%s, %s) → %s\n"
        "⏳ This may take a while depending on model size. Please do not interrupt.",
        spec.kind,
        spec.model_name,
        spec.source_model,
        f", {spec.weight_format}" if spec.weight_format else "",
        spec.target_device,
        spec.model_dir,
    )
    # ModelScope pulls: export_model.py honors HF by default; the source id + hub env
    # decide where weights come from (mirrors models.model_hub).
    env_hub = _hub()
    if env_hub in ("modelscope", "ms"):
        os.environ.setdefault("USE_MODELSCOPE", "1")

    rc = run_cli(cmd=cmd, log_fn=logger.info)
    if rc != 0:
        raise RuntimeError(
            f"OVMS export of '{spec.model_name}' ({spec.kind}) failed with exit code "
            f"{rc}. See the export log above for the cause."
        )
    if not spec.is_ready():
        raise RuntimeError(
            f"OVMS export of '{spec.model_name}' reported success but no IR/graph was "
            f"written under {spec.model_dir}."
        )
    logger.info("✅ Exported servable '%s'", spec.model_name)


def write_config_json(specs: List[Servable]) -> Path:
    """Write the canonical ``config.json`` listing the three mediapipe servables."""
    doc = {
        "model_config_list": [],
        "mediapipe_config_list": [
            {"name": s.model_name, "graph_path": s.graph_path} for s in specs
        ],
    }
    OVMS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OVMS_CONFIG_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "✅ Wrote %s (%d servables: %s)",
        OVMS_CONFIG_PATH,
        len(specs),
        ", ".join(s.model_name for s in specs),
    )
    return OVMS_CONFIG_PATH


def provision_ovms_models(
    specs: Optional[List[Servable]] = None,
    force: bool = False,
    config_only: bool = False,
) -> Path:
    """Provision ``models/ovms/`` end to end and return the repo directory."""
    specs = specs or build_specs()

    if not config_only:
        script = ensure_export_script()
        for spec in specs:
            export_servable(script, spec, force=force)

    write_config_json(specs)
    logger.info("🎉 OVMS model repository ready at %s", OVMS_MODELS_DIR)
    return OVMS_MODELS_DIR


def _selected_specs(args: argparse.Namespace) -> List[Servable]:
    all_specs = {"text_gen": _text_gen_spec, "embeddings": _embeddings_spec, "rerank": _rerank_spec}
    chosen = [k for k in all_specs if getattr(args, k)]
    if not chosen:  # none flagged => all
        chosen = list(all_specs)
    return [all_specs[k]() for k in chosen]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Provision the OVMS model repository (0.C.1).")
    parser.add_argument("--force", action="store_true", help="Re-export even if IR exists.")
    parser.add_argument("--config-only", action="store_true", help="Only (re)write config.json.")
    parser.add_argument("--text-gen", dest="text_gen", action="store_true", help="Provision only text_gen.")
    parser.add_argument("--embeddings", action="store_true", help="Provision only embeddings.")
    parser.add_argument("--rerank", action="store_true", help="Provision only rerank.")
    args = parser.parse_args()

    provision_ovms_models(
        specs=_selected_specs(args),
        force=args.force,
        config_only=args.config_only,
    )


if __name__ == "__main__":
    main()
