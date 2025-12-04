"""Hyperparameter sweep runner for latent data distillation experiments.

This script orchestrates multiple calls to `main.py` with varied hyperparameters
and aggregates the resulting metrics and distillation artifacts. It is designed
for reproducible benchmarking: every run uses a named subdirectory so that
reports, prototype images, and distilled bundles stay isolated.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from config import (
    AE_EPOCHS,
    DEFAULT_AE_BATCH_SIZE,
    DEFAULT_BACKBONE,
    DEFAULT_DATASET,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_WORKERS,
    MAX_TRAIN_POINTS,
    M_PER_CLASS,
    PROTO_RL_LR,
    RESULTS_DIR,
    SEED,
    SEL_BUDGET_PER_CLASS,
)
from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run a grid of latent DD experiments")
    parser.add_argument("--datasets", nargs="+", default=[DEFAULT_DATASET])
    parser.add_argument("--backbones", nargs="+", default=[DEFAULT_BACKBONE])
    parser.add_argument("--budgets", nargs="+", type=int, default=[SEL_BUDGET_PER_CLASS], help="per-class selection budgets")
    parser.add_argument("--prototypes", nargs="+", type=int, default=[M_PER_CLASS], help="prototypes per class")
    parser.add_argument("--ae-epochs", nargs="+", type=int, default=[AE_EPOCHS])
    parser.add_argument("--proto-rl-lrs", nargs="+", type=float, default=[PROTO_RL_LR])
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--name", default="sweep", help="Name for the sweep folder under results/")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--custom-train-dir", default=None)
    parser.add_argument("--custom-test-dir", default=None)
    parser.add_argument("--override-image-size", type=int, default=None)
    parser.add_argument("--max-train-points", type=int, default=MAX_TRAIN_POINTS)
    parser.add_argument("--ae-batch-size", type=int, default=DEFAULT_AE_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def slugify(value: str) -> str:
    return value.replace("/", "-").replace(" ", "_").replace(".", "p")


def build_command(
    dataset: str,
    backbone: str,
    budget: int,
    m_per_class: int,
    seed: int,
    ae_epochs: int,
    proto_rl_lr: float,
    args,
    run_name: str,
    sweep_root: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "RL_latentDD.main",
        f"--dataset={dataset}",
        f"--backbone={backbone}",
        f"--sel-budget-per-class={budget}",
        f"--m-per-class={m_per_class}",
        f"--seed={seed}",
        f"--ae-epochs={ae_epochs}",
        f"--proto-rl-lr={proto_rl_lr}",
        f"--results-dir={sweep_root}",
        f"--run-name={run_name}",
        f"--ae-batch-size={args.ae_batch_size}",
        f"--num-workers={args.num_workers}",
        f"--max-train-points={args.max_train_points}",
        f"--data-root={args.data_root}",
    ]
    if args.custom_train_dir:
        cmd.append(f"--custom-train-dir={args.custom_train_dir}")
    if args.custom_test_dir:
        cmd.append(f"--custom-test-dir={args.custom_test_dir}")
    if args.override_image_size:
        cmd.append(f"--override-image-size={args.override_image_size}")
    return cmd


def main():
    args = parse_args()
    sweep_root = ensure_dir(Path(args.results_dir) / args.name)

    combos = list(
        itertools.product(
            args.datasets, args.backbones, args.budgets, args.prototypes, args.seeds, args.ae_epochs, args.proto_rl_lrs
        )
    )

    summary = []
    for dataset, backbone, budget, m_pc, seed, ae_epochs, proto_rl_lr in combos:
        run_name = slugify(f"{args.name}_d-{dataset}_b-{backbone}_bud-{budget}_mpc-{m_pc}_seed-{seed}_ae-{ae_epochs}_prl-{proto_rl_lr}")
        cmd = build_command(dataset, backbone, budget, m_pc, seed, ae_epochs, proto_rl_lr, args, run_name, sweep_root)

        if args.dry_run:
            print("[DRY-RUN]", " ".join(cmd))
            continue

        print("\n[SWEEP] Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        run_dir = sweep_root / dataset / backbone / run_name
        report_path = run_dir / "experiment_summary.json"
        distill_path = run_dir / "distillation_artifacts.pt"

        report_payload = {}
        if report_path.exists():
            with report_path.open() as f:
                report_payload = json.load(f)
        else:
            print(f"Warning: report missing at {report_path}")

        summary.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "budget_per_class": budget,
                "m_per_class": m_pc,
                "seed": seed,
                "ae_epochs": ae_epochs,
                "proto_rl_lr": proto_rl_lr,
                "run_dir": str(run_dir),
                "report": report_payload,
                "distillation_artifacts": str(distill_path),
            }
        )

    if not args.dry_run and summary:
        sweep_summary_path = sweep_root / "sweep_summary.json"
        with sweep_summary_path.open("w", encoding="utf-8") as f:
            json.dump({"runs": summary}, f, indent=2)
        print(f"\nSweep complete. Aggregated summary written to {sweep_summary_path}")


if __name__ == "__main__":
    main()
