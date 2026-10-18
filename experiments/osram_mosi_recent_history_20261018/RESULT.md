# Short-window history residual: MOSI seed66

INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle W-F1.

Status: complete,150epochs. Verified remote SUMMARY.json and PROVENANCE.json;
no new training or inference was launched to obtain this report.

| Model | Seed | 8-rate W-F1 | High-missing W-F1 |
|---|---:|---:|---:|
| Original Flat150 |66|81.068095|76.352251|
| Flat + strict short-window residual |66|80.480858|75.844614|
| Difference (percentage points) |66|−0.587237|−0.507637|

The requested80.7 reference is the original Flat150 **three-seed mean80.747**,
not the seed66 baseline. Compare this new single run with the seed66 row above.

Only the extra W4 observed-latent attention branch is added before the original
Flat LayerNorm. No CHM write/read equation, Query, K/V, Local Skip, classifier,
missing schedule or task loss is changed. Added parameters370,624.

37 automated tests and a separate real-data GPU smoke passed. Zero-init
predictions/RNG and original mask hashes matched; these checks establish
implementation correctness, not task effectiveness. Launch, runtime, immutable
source and outputs are recorded in `LAUNCH.md`. Completed remote output:
`/home/qinjinghui/remote_experiments/osram_mosi_recent_history_20261018/seed66_v1`.
`SUMMARY.json` contains all eight rates, exact matching mask hashes, selected
epochs and paired original-baseline differences. Eight-rate ACC80.564024;
high-missing ACC75.965447. This single-seed residual did not improve the matched
original Flat150, so no additional runs are authorized by this report.

| Missing rate | ACC | W-F1 | Selected epoch |
|---|---:|---:|---:|
|0.0|87.804878|87.678071|72|
|0.1|85.975610|85.852135|72|
|0.2|82.621951|82.595054|71|
|0.3|80.945122|80.940446|71|
|0.4|79.268293|79.247319|98|
|0.5|76.676829|76.621090|70|
|0.6|75.000000|74.902610|60|
|0.7|76.219512|76.010142|57|
