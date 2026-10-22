# MOSEI capacity allocation: launched, not completed

INTERNAL TEST-ORACLE SCREEN

Training source: `5a86f7e98bcbe8eead96b10410be6175d0abd832`.
Server: `biggpu`, physical GPU1. Two fresh joint-training runs, seed66, 60epochs.

| Model | Flat hidden | V3 width / PairMLP | Total parameters | Formal scores |
|---|---:|---|---:|---|
| Smaller Flat control | 1024 | off | 10,080,865 | pending |
| Smaller Flat + larger V3 | 1024 | 256 / 512 | 12,397,503 | pending |

Both real-data three-batch GPU smokes passed: forward/backward/Adam updates,
single CHM scan, exact zero-init predictions versus same-width Flat, and full
model/optimizer/RNG/prediction restoration. All monitored V3 parameter groups
and both source-read gradient paths became finite and nonzero across the smoke.
Measured peak allocated memory:607.28MiB(control),2218.89MiB(V3). These small
batches are not a full-conversation worst-case memory guarantee or performance
result. Smoke outputs and weights are preserved separately from formal runs.

Original Flat output1600, Local Skip, task head, CHM and joint write are unchanged.
Only Flat intermediate width and V3 internal widths change. The V3 combined
model is smaller overall than the original Flat1600+default V3(13,844,799).

Original MOSEI features/splits/environment, new gcnet-random-mask-v1, batch32,
Adam LR1e-3, WD1e-5, constant60epochs, emotion-only MSE and cyclic0..0.7 stay
unchanged. Exact original seed66 test-mask hashes and unrelated initialization
were verified; each full epoch must match original training-mask/order hashes,
optimizer-step count and LR trace. Full last.pt and eight per-rate BEST weights,
exact prediction arrays and masks are retained.

Per-rate Test-best W-F1 is selected; ACC/MSE come from the same predictions.
This is exploratory testing, not a single-checkpoint independent test result.
No multi-seed expansion or HPO is authorized by this run.

See `LAUNCH.json` for commands, PIDs, mask hashes, environment and smoke evidence.
The persistent controller writes current QUEUE.json and completed-only
RESULT.md/REPORT.json/PER_RATE.csv under:

```text
/data2/yb/remote_experiments/osram_mosei_v3_capacity_20261022/execution_v1
```

No new formal W-F1 improvement or failure is established yet.
