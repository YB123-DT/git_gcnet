# rho.10 task-only: fixed whole-conversation missing

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY

MOSI seed66; frozen existing models, no training or new checkpoint selection.
Each conversation keeps only the named modalities throughout. T=Text.
Single modality: miss=.7 BEST; pairs: miss=.3 BEST. Original fixed evaluator.
686 test utterances,656 nonneutral labels; threshold prediction>0, weighted binary F1.
Historical backbone6e2a998 source hashes match original rho.10 C02. No training projector calls.
Flat/rho.50 reuse seed66 records, not three-/five-seed averages. Current rho.10+InfoNCE training is separate.

| Visible | Flat | rho.50 task-only | rho.10 task-only | rho.10−Flat pp |
|---|---:|---:|---:|---:|
| A | 35.643 | 56.986 | 25.404 | -10.239 |
| T | 85.620 | 84.540 | 86.211 | +0.591 |
| V | 53.381 | 60.264 | 52.071 | -1.310 |
| AT | 86.478 | 84.074 | 86.334 | -0.144 |
| AV | 57.958 | 58.661 | 59.118 | +1.160 |
| TV | 86.754 | 84.974 | 86.026 | -0.728 |
| No Text | 48.994 | 58.637 | 45.531 | -3.463 |
| Text present | 86.284 | 84.529 | 86.190 | -0.094 |
| Six-pattern mean | 67.639 | 71.583 | 65.861 | -1.778 |

Units W-F1%; differences pp. Six patterns equally averaged, not a naturally weighted test population.
Single seed. Different trained models and selected epochs; no causal mechanism or stable multi-seed benefit claim.
No threshold tuning, synthetic modality completion, new masks for training or deployment performance claim.

Runner commit: ec4fd9e; server biggpu GPU6. Command:

```bash
python -m experiments.osram_paired_rho010_infonce_20261011.fixed \
  --source /data2/yb/remote_experiments/osram_paired_rho_sweep_20261002/runs/C02 \
  --output <new-output.json> --runner-commit ec4fd9e
```
