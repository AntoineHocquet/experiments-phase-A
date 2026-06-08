"""Run every kept Phase A proof-of-concept experiment in sequence.

Usage:
  python run_all.py --smoke    # tiny sizes on CPU; a quick end-to-end check
  python run_all.py            # full sizes (hours; benefits from a GPU)

In --smoke mode the Wave 2 scripts take their own --smoke flag; the four
foundational scripts (run_e2b, run_e3a, run_e3b, run_e3b_2d) predate that flag
and instead receive explicit small sizes. Each experiment writes into its own
results_<exp>/ directory (gitignored).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (script, out_dir, smoke args, full args). Smoke args already encode tiny sizes;
# full args are empty so each script uses its committed defaults.
EXPERIMENTS = [
    # Pillar 1: operator splitting
    ("commutator/run_e2b.py", "results_e2b",
     ["--n_pretrain", "80", "--n_pretrain_epochs", "4", "--ns", "80",
      "--n_finetune_epochs", "4", "--n_seeds", "1", "--n_test", "16",
      "--nx", "32", "--width", "16", "--n_modes", "8", "--n_layers", "2"], []),
    ("commutator/run_e2d.py", "results_e2d", ["--smoke"], []),
    ("commutator/run_e2e.py", "results_e2e", ["--smoke"], []),
    # Pillar 2: geometry-adaptive spectral basis
    ("geometry/run_e3a.py", "results_e3a",
     ["--n_train", "120", "--n_epochs", "25"], []),
    ("geometry/run_e3b.py", "results_e3b",
     ["--nx", "32", "--n_train", "80", "--n_epochs", "15",
      "--enc_train", "120", "--enc_epochs", "25"], []),
    ("geometry/run_e3b_2d.py", "results_e3b_2d",
     ["--nx", "16", "--ny", "16", "--n_train", "60", "--n_epochs", "8",
      "--enc_train", "120", "--enc_epochs", "25", "--n_test", "6"], []),
    ("geometry/run_e3d.py", "results_e3d", ["--smoke"], []),
    ("geometry/run_e3f.py", "results_e3f", ["--smoke"], []),
    ("geometry/run_e3g.py", "results_e3g", ["--smoke"], []),
    ("geometry/run_e3h.py", "results_e3h", ["--smoke"], []),
    ("geometry/run_e3i.py", "results_e3i", ["--smoke"], []),
    # Pillar 3: domain decomposition
    ("schwarz/run_e4e.py", "results_e4e", ["--smoke"], []),
    # Pillar 1 x Pillar 2 bridge
    ("commutator_geometry/run_e5a.py", "results_e5a", ["--smoke"], []),
]


def run(script: Path, args: list) -> int:
    cmd = [sys.executable, str(script)] + args
    print(f"\n{'=' * 60}\nRunning: {script.name}  {' '.join(args)}\n{'=' * 60}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny sizes on CPU for a quick end-to-end check.")
    args = ap.parse_args()

    codes = []
    for script, out_dir, smoke_args, full_args in EXPERIMENTS:
        extra = (smoke_args + ["--device", "cpu"]) if args.smoke else full_args
        extra = extra + ["--out_dir", out_dir]
        codes.append(run(ROOT / script, extra))

    any_fail = any(c != 0 for c in codes)
    print(f"\n{'=' * 60}\nAll experiments {'PASSED' if not any_fail else 'SOME FAILED'}\n{'=' * 60}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
