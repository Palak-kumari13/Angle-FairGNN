# Optional FRME research track (separate from AngleFair)

This folder preserves the previously supplied centralized FRME v2.4 graph-reconstruction bridge, predictor, explainer and retention prototype. It is not secretly merged into AngleFair's gradient/geometry training, and its `run_integrated.py` remains a synthetic-only prototype. Its learned reconstruction weights are not yet fully connected into downstream gate selection. No federated-learning code is included.

From the repository root:

```bash
cd optional_frme
python run_integrated.py --epochs 10 --seeds 1 --out results/frme_smoke.json
```

If graph construction fails, the optional route can choose features. In AngleFair, A0 is the reference fallback. These are distinct controllers; do not claim a single jointly trained end-to-end system.
