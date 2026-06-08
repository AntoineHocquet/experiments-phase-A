"""Shared utilities for the Phase A experiments.

Every script in commutator/, geometry/, schwarz/ and the cross-pillar folders
follows the same conventions:

  1. a `--device {auto,cuda,cpu}` flag (auto picks cuda if available);
  2. small CPU-friendly defaults so a smoke test runs in 1 to 3 minutes on a
     laptop, but a `--smoke` flag that shrinks them further to about 30 seconds;
  3. one JSON dump per run (compute time, key headline numbers, parameters)
     and one PDF + PNG plot in the same out_dir.

This module collects the common helpers.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def auto_device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def add_device_args(ap: argparse.ArgumentParser) -> None:
    """Add the standard --device / --smoke / --seed / --out_dir flags."""
    ap.add_argument("--device", default="auto",
                    help="'auto' (cuda if available), 'cuda', or 'cpu'.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny CPU smoke test (a few tens of seconds).")
    ap.add_argument("--seed", type=int, default=0)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class Timer:
    """Wall-time accumulator with sub-section labels.

    Usage:
        timer = Timer()
        with timer("train"):
            ...
        with timer("eval"):
            ...
        timer.dump()  # returns dict with per-section and total seconds
    """

    def __init__(self) -> None:
        self._sections: dict[str, float] = {}
        self._t0: float = time.perf_counter()

    @contextmanager
    def __call__(self, name: str):
        t = time.perf_counter()
        yield
        self._sections[name] = self._sections.get(name, 0.0) + (time.perf_counter() - t)

    def dump(self) -> dict[str, float]:
        total = time.perf_counter() - self._t0
        out = {"total_seconds": total}
        for name, sec in self._sections.items():
            out[f"{name}_seconds"] = sec
        return out


# ---------------------------------------------------------------------------
# Run metadata dump
# ---------------------------------------------------------------------------

def write_run_json(out_dir: Path | str, *,
                   experiment: str,
                   pillar: str,
                   hypothesis: str,
                   parameters: dict[str, Any],
                   headline: dict[str, Any],
                   timing: dict[str, float],
                   device: str,
                   extra: dict[str, Any] | None = None) -> Path:
    """Write the canonical results_<experiment>.json file.

    The schema is deliberately small and human-readable so the JSON can be
    grepped and diffed in the reports/ folder without parsing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "pillar": pillar,
        "hypothesis": hypothesis,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else None),
        "platform": platform.platform(),
        "parameters": parameters,
        "headline": headline,
        "timing": timing,
    }
    if extra is not None:
        payload["extra"] = extra
    path = out_dir / f"{experiment}_raw.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return path


def _json_default(obj: Any) -> Any:
    """Tolerant JSON encoder for numpy / torch scalars."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Plot helpers (PDF + PNG, consistent styling)
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, out_dir: Path | str, name: str) -> tuple[Path, Path]:
    """Save the same figure as both PDF (vector) and PNG (raster preview)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Convenience: print headline summary
# ---------------------------------------------------------------------------

def print_summary(experiment: str, headline: dict[str, Any],
                  timing: dict[str, float]) -> None:
    width = 65
    print(f"\n{'═' * width}")
    print(f"  SUMMARY: {experiment}")
    print(f"{'═' * width}")
    for k, v in headline.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>20.6g}")
        else:
            print(f"  {k:<35} {str(v):>20}")
    print(f"  {'wall time (s)':<35} {timing.get('total_seconds', 0.0):>20.1f}")
    print(f"{'═' * width}\n")
