# AngleFair-GNN

**Research software, not proven to outperform published methods.** The default AngleFair track needs no FRME module. The Global Fallback Controller is enabled in `train.py` and chooses checkpoints using validation-only utility, DP/EO and drift. A failed mask is **not reported as a faithful explanation**.

## AngleFair core

A0 original GCN; A1 mask; A2 mask/fidelity; A3 train-only fairness; **A4** extends A3 with perturbed-graph prediction consistency and aligned-mask stability; **A5** extends A4 with extra frozen-A0 teacher KL. A4/A5 are implemented ablations, not demonstrated solutions to previous mask collapse. Geometry `diagnose.py` remains exploratory and must not be interpreted on collapsed masks. These A4/A5 stages are not continual-learning benchmarks: the teacher is trained on the same task.

### Install / test

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --data-dir data --dataset credit --stage 4 --seeds 0 1 2 3 4 --epochs 150 --output results/credit_stage4
python diagnose.py --data data/credit.npz --checkpoint results/credit_stage4/credit_A3_seed0.pt --proxy --output results/credit_angles
```

Input NPZ: `x`, `edge_index`, `y`, `sensitive`, `train_idx`, `val_idx`, `test_idx`; binary labels and sensitive groups, fixed and disjoint splits. Test metrics are reporting-only. Output includes checkpoints, per-seed CSV, and `fallback_seedN.json`.

### Synthetic runnable example

```bash
python generate_synthetic.py --angles 30 --seeds 0
python run_sweep.py --angles 30 --seeds 0 --stage 4 --epochs 20 --output results/smoke
```

### Optional separate FRME Graph Reconstruction Bridge and controller

See [`optional_frme/README.md`](optional_frme/README.md). It contains separately runnable reconstructed-graph / fallback / fairness and stability / retention experiments. **Not a jointly trained AngleFair graph bridge**; all tested numerical gains must be established independently. It is excluded from AngleFair's default runner.

## Prior results that govern interpretation

- Bail feature-only MLP AUC ~0.992 vs original GCN ~0.931: evaluate leakage and do not force graph explanations.
- Credit original GCN AUC ~0.707 vs MLP ~0.647: prioritize structural signal verification.
- A1–A5 masks previously collapsed often, and Credit masked improvements over random were negative in the 90-run sweep.
- Exact top-k budgets don't guarantee an informative ranking; require planted-edge recovery, mask score spread, permutation invariance, and equal-budget random/degree controls.
- Fairness, stability and forgetting are separate empirical claims; passing a synthetic smoke test does not validate them on Bail, Credit or German.

The supplied report remains the source of historical observations. No benchmark improvement is claimed by these code additions.

## Verified local smoke run (synthetic only)

Executed stage 4 with 20 epochs, angle=30, seed=0: A0–A5 completed, global fallback selected A3 on validation and **abstained from certifying the explanation**. See `examples/synthetic_fallback_seed0.json`. The optional FRME test suite can be run from `optional_frme/` using `python -m unittest discover -s tests -q`. No results on real benchmarks are implied.
