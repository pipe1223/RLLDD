# RL-guided distilled dataset parametrization in latent feature space.

This project explores **dataset distillation in latent space** combined with **reinforcement learning (RL)**.  
The aim is to study whether **RL can help shape synthetic prototypes** in a latent feature space to match or beat strong baselines.

---

## High-Level Idea

1. Train a **convolutional autoencoder with a classifier head** on CIFAR-10.
   - Encoder → latent representation `z`
   - Decoder → reconstructs the image from `z`
   - Classifier head → predicts label from `z`

2. Perform all dataset distillation **in latent space**:
   - Learn **synthetic latent prototypes** per class.
   - Train a **linear classifier** on these prototypes and evaluate on real test latents.

3. Add **Reinforcement Learning** in two places:
   - **RL-Selection**: RL chooses which real latent points to keep under a per-class budget.
   - **RL-Shaping of prototypes (Actor–Critic)**: RL controls the loss weighting schedule during prototype training, starting from a strong baseline (fine-tuning).

4. Decode prototypes back to image space for **visualization & intuition**.

This gives you a full experimental pipeline for **latent + RL + dataset distillation (DD)**.

### Reproducibility

- All runs now enable deterministic PyTorch flags via `set_seed`, which also seeds NumPy and CUDA RNGs.
- A structured JSON report (`results/experiment_summary.json`) is written at the end of `main.py` runs, capturing the full configuration snapshot and core metrics for transparent, publication-grade tracking.

### Backbone and dataset choices

`main.py` now accepts CLI flags so you can swap in stronger encoders or different datasets without editing code:

```
python RL_latentDD/main.py --dataset cifar100 --backbone resnet18
python RL_latentDD/main.py --dataset custom --custom-train-dir /path/to/train --custom-test-dir /path/to/test --backbone vit_b_16
```

- `--dataset` supports `cifar10`, `cifar100`, or `custom` (ImageFolder directories).
- `--backbone` supports the original `conv` autoencoder, `resnet18`, or `vit_b_16` for stronger latent encoders.
- For custom datasets, optional `--override-image-size` lets you control the resize applied during preprocessing.

### Research-grade experiment harness

- All experiment knobs (data split, backbone, loader workers, AE batch size) are captured in structured dataclasses and printed at run start for auditable provenance.
- Results and artifacts are now organized under `results/<dataset>/<backbone>/` so multiple backbones/datasets can be benchmarked side by side without collisions.
- Safety checks prevent impossible configurations (e.g., too-small training splits or selection budgets larger than the latent pool), surfacing actionable errors instead of silent failures.

### Hyperparameterized single runs

The main script exposes the core research levers directly on the CLI so you can run controlled ablations without code edits:

```
python RL_latentDD/main.py \
  --dataset cifar10 \
  --backbone vit_b_16 \
  --seed 123 \
  --ae-epochs 20 \
  --latent-dim 256 \
  --sel-budget-per-class 15 \
  --m-per-class 15 \
  --proto-steps-base 500 \
  --proto-rl-lr 5e-5 \
  --run-name vit_run_seed123
```

Every run creates its own timestamped (or `--run-name`) directory containing:

- `experiment_summary.json` with configuration + metrics (including accuracy of baselines, RL selection, and RL-shaped prototypes).
- Prototype, selection, and t-SNE visualizations saved next to the report for quick inspection.
- `distillation_artifacts.pt` bundling the selected latent indices, latent tensors, labels, and prototype state dicts for downstream analysis or reuse.

### Automated sweeps with aggregation

Use `RL_latentDD/sweep.py` to launch a grid of experiments and collect their outputs:

```
python RL_latentDD/sweep.py \
  --datasets cifar10 cifar100 \
  --backbones conv resnet18 \
  --budgets 10 20 \
  --prototypes 10 20 \
  --proto-rl-lrs 1e-4 5e-5 \
  --seeds 0 1 \
  --name backbone_budget_sweep
```

The sweep writes a `sweep_summary.json` aggregating all per-run reports and pointing to their distillation bundles under `results/<name>/<dataset>/<backbone>/<run_name>/`.

---
