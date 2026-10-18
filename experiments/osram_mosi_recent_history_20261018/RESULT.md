# Short-window history residual: MOSI seed66

INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle W-F1.

Status: launched and verified running on biggpu physical GPU3. This is not a
completed150-epoch result. Final scores remain pending.

| Model | Seed | 8-rate W-F1 | High-missing W-F1 |
|---|---:|---:|---:|
| Original Flat150 |66|81.068095|76.352251|
| Flat + strict short-window residual |66|pending|pending|

The requested80.7 reference is the original Flat150 **three-seed mean80.747**,
not the seed66 baseline. Compare this new single run with the seed66 row above.

Only the extra W4 observed-latent attention branch is added before the original
Flat LayerNorm. No CHM write/read equation, Query, K/V, Local Skip, classifier,
missing schedule or task loss is changed. Added parameters370,624.

37 automated tests and a separate real-data GPU smoke passed. Zero-init
predictions/RNG and original mask hashes matched; these checks establish
implementation correctness, not task effectiveness. Launch, runtime, immutable
source and outputs are recorded in `LAUNCH.md`. The finite runner automatically
produces `SUMMARY.json` with all eight rates, overall/high-missing metrics and
paired original-baseline differences when stage150 finishes.
