# Neural Production Local correction: wider rule MLP

INTERNAL DIAGNOSTIC ONLY

One fixed capacity comparison requested by the user; not a new mechanism.
Only the four shared-over-steps rule MLPs change from 128->96->64 to
128->256->64. Keep 4 rules, 3 steps, 64-d tokens, 8 memory heads,
Local/Base/Gap zero-initialized output bridges and original Local Skip.
Memory, query, writes, task head and loss unchanged.

Module parameters: 191808 -> 315328 (+123520, +64.4%).
These are module counts, not whole-model counts.
Old default remains 96. Existing checkpoints keep their old parameter shapes.
New width is a from-scratch experiment, NOT checkpoint continuation.

MOSI seed66, 100 epochs, original cfg84 no-JEPA cyclic random-missing
protocol, same optimizer/lr/batch, eight per-rate BEST checkpoints.
Test-oracle scores are internal screening, not formal paper results.
No automatic multi-seed expansion. No auxiliary loss or additional view.

Implementation: NeuralProduction(rule_hidden), variant neural_production_local_w256,
existing core20 runner and NPS dispatcher (--method). Existing small run untouched.
Plan: bounded CPU shape/mask/gradient check, isolated git snapshot, GPU admission,
launch one run on biggpu GPU5; verify process and effective configuration.

Root: /data2/yb/remote_experiments/osram_nps_local_w256_20261009
Command (from sealed source):

```text
python -m experiments.osram_nps_local_20261009.dispatch
 --root /data2/yb/remote_experiments/osram_nps_local_w256_20261009
 --method neural_production_local_w256
 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
 --gpu-index 5 --gpu-uuid GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62
```

Verification passed on biggpu CPU: width/parameter count, zero-init output
parity, rule weights update with finite gradients, inactive NaN exclusion,
padding evidence zero; old Local variant RNG/parameter parity test passed.
Local syntax compilation and git diff whitespace checks passed.
Initial check failed as expected because the new variant did not exist.
Status: RUNNING, no final result yet.
Source snapshot: 1bc4c36. Started 2026-10-09T03:11:12.599102+00:00.
biggpu physical GPU5, UUID as above; free 11671 MiB at admission.
tmux nps_local_w256_20261009; dispatcher PID3460071; training PID3460257.
Independent source/config/output, original small run not stopped or modified.
Log: root/seed_66/train.log; state: root/DISPATCH.json.

## Completed seed66 result

Status COMPLETE supersedes RUNNING: 100/100 epochs, exit_code=0,
outputs_verified=true. Rechecked all 20 recorded artifact SHA256 hashes.
Eight per-rate BEST checkpoints/predictions and last_training.pt retained.
No new inference or retraining for this summary. ACC is measured at the
W-F1-selected checkpoint for each rate, not separately selected on ACC.

| Missing rate | BEST epoch | ACC (%) | W-F1 (%) |
|---|---:|---:|---:|
| 0.0 | 59 | 87.195122 | 87.195122 |
| 0.1 | 61 | 85.823171 | 85.851310 |
| 0.2 | 59 | 82.317073 | 82.385532 |
| 0.3 | 62 | 80.487805 | 80.468065 |
| 0.4 | 57 | 80.182927 | 80.052647 |
| 0.5 | 80 | 75.914634 | 76.048976 |
| 0.6 | 62 | 76.067073 | 76.083976 |
| 0.7 | 57 | 74.847561 | 74.654761 |

Equal-rate mean ACC/W-F1: 80.354421 / 80.342549.
High-missing (.5/.6/.7) mean ACC/W-F1: 75.609756 / 75.595904.

| Seed66 version | Mean8 W-F1 | High W-F1 |
|---|---:|---:|
| Original NPS, Base/Gap correction only | 80.981287 | 76.475237 |
| NPS plus Local correction, rule width96 | 80.442790 | 75.647545 |
| NPS plus Local correction, rule width256 | 80.342549 | 75.595904 |

W256 minus W96: -0.100242 pp mean8, -0.051641 pp high.
W256 minus original NPS: -0.638738 pp mean8, -0.879333 pp high.
This single-seed comparison provides no gain from the tested width increase;
it does not establish that all larger NPS variants are ineffective. Do not
automatically expand to more seeds or capacity settings. No mechanism claim.
INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle, not a formal paper result.
