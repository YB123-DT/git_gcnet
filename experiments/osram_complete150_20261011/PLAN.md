# Complete150: finite62-run batch

INTERNAL DIAGNOSTIC ONLY. User explicitly prioritized items1/2; no paired-view
jobs, new modules, gates, auxiliary losses or hyperparameter searches.

- MOSI Flat67/68:2fresh100epoch runs then50full-state continuation.
- IEMOCAPFour/Six Flat66/67/68 x5folds:30fresh100 then50continuation.
- IEMOCAPFour/Six original Nested66/67/68 x5folds:30existing full100 states
  continued for50epochs. No original states or results overwritten.

Budget100atLR1e-3 then50at1e-4. This is NOT BEST-only initialization or
constantLR150. Flat histories lack recovery states; their newly trained100
results must be reported separately from archived Flat100, not assumed identical.
Original dataset/class/fold/seed configs and sealed historical sources retained.
IEMOCAP selects ACC and reports ACC/UA; MOSI selects W-F1 and reports ACC/W-F1.
Per-rate cumulative BEST includes original100. Such expanded selection cannot
lower the selection metric; it does not alone prove generalization improvement.

All execution stays on biggpu. GPUs0/1/2/3/7 only, maximum2jobs each with9GB
reservation and actual free-memory admission. GPU4 forbidden; busy5/6 excluded.
Independent task logs/configs/metrics/predictions/BEST/full-state retained.
Use same-host /home/qinjinghui/yb_gcnet_experiments/osram_complete150_20261011
on root filesystem after capacity verification: data2's94GiB is insufficient
for conservative62-run recovery/version budgets. No migration of existing data.

Preparation verifies pinned source hashes, data contents, exact configs and
all30original recovery hashes. Children check source hashes and data metadata,
record effective configs and actual per-epoch optimizer LRs. Full restore uses
model, Adam state, RNG, sampler/mask identity and BEST selection state. Copy only
immutable versions referenced by the source recovery state; leave source intact.
Final checks:150history entries, original100prefix,50newLR traces, unchanged
evaluation masks, correct selection metric,20required artifacts and unchanged
source recovery hash. Failure halts new admissions and drains running jobs.

Commands: `python run.py --root ROOT --prepare --commit COMMIT`, then persistent
`python -u run.py --root ROOT --dispatch`. Source runner is copied after commit
and never edited while jobs run. No long smoke training; targeted configuration
and optimizer-state tests plus actual first-epoch checks.
