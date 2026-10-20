# Launch verified: two MOSI150 runs in progress

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY

Observed2026-10-20T10:48:16Z: Flat PID2119203 and Router PID2119428 both
completed epoch1 and remained running on biggpu physicalGPU2. Controller
PID2116333 is persistent under tmux socket`osram_mosi_lr2e5`, session
`mosi_lr2e5_f95f28a`. This is a launch snapshot, not final performance.

Both effective configs and actual optimizer logs confirm backbone2e-5,
batch32,wd1e-5,150epochs,GCNet random missing and per-rateTest-bestW-F1.
Router remains3e-4 for1–100,3e-5 for101–150. Original initialization,
data split, model/sampler numerical core and exact evaluation masks match
the completed original seed66 reference. Epoch1 training masks and update
counts matched; later epochs are checked automatically, not yet claimed.

24focused local tests passed; compileall and diff checks passed. Existing
optional torch-sparse warning was nonfatal. The short real GPU smoke passed
finite/nonzero gradients, actual backbone/Router updates, full model/Adam/RNG
reload and exact selected-prediction replay. Peak3092.058MiB. Smoke is not
evidence of improved performance; only two Router parameter tensors initially
had nonzero gradients, so connection is not misreported as nonzero gradients.

Frozen source`f95f28aef9e02cbf4f6c8ebd78787feb4bdc3887`; base+delta hashes,
configuration and initialization checks are in LAUNCH.json. All six changed
runner/wrapper file hashes matched between local commit and remote snapshot.
Output:`/data2/yb/remote_experiments/osram_mosi_router_lr2e5_20261020/execution_v1`.

Queue saves eight BEST weights and matched predictions per run plus complete
last recovery state, and writes final RESULT.md/METRICS.csv/SUMMARY.json only
after both150epoch trajectories complete and prediction checks pass. No old
weights/results were overwritten, no extra seeds or datasets were added.
