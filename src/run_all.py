"""
run_all.py — Convenience script to run the full pipeline end-to-end.

Executes in order:
  1. data_pipeline.py   — raw CSV → processed tensors
  2. baselines.py       — persistence + LSTM baselines
  3. train.py           — Transformer (full ablation matrix)
  4. evaluate.py        — metrics, ablation table, plots
  5. Generates RESULTS_SUMMARY.md from final metrics

Usage:
    python src/run_all.py
    python src/run_all.py --no-ablations   # train only default config, faster
"""

import sys
import subprocess
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(cmd: list[str], label: str):
    log.info("=" * 70)
    log.info("STEP: %s", label)
    log.info("CMD:  %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("Step '%s' failed with return code %d. Check output above.", label, result.returncode)
        sys.exit(result.returncode)
    log.info("Step '%s' complete.", label)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ablations", action="store_true",
                        help="Skip full ablation matrix (only run default config).")
    args = parser.parse_args()

    py = sys.executable
    src = Path(__file__).resolve().parent

    # Step 1 — Data pipeline
    run([py, str(src / "data_pipeline.py")], "Data Pipeline")

    # Step 2 — Baselines
    run([py, str(src / "baselines.py"), "--all-states"], "Baselines")

    if args.no_ablations:
        # Step 3 — Train default config only
        run([py, str(src / "train.py"), "--all-states"], "Training (default config)")
        # Step 4 — Evaluate default
        run([py, str(src / "evaluate.py"), "--all-states"], "Evaluation")
    else:
        # Step 3+4 — Full ablation (train + evaluate all configs)
        run(
            [py, str(src / "evaluate.py"), "--all-states", "--run-ablations"],
            "Full Ablation Training + Evaluation",
        )

    # Step 5 — Generate RESULTS_SUMMARY.md
    run([py, str(src / "generate_summary.py")], "Results Summary Generation")

    log.info("=" * 70)
    log.info("All steps complete. See results/ directory.")


if __name__ == "__main__":
    main()
