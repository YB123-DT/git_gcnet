# V3 three-seed completion

INTERNAL TEST-ORACLE SCREEN

User correction: use seeds66/67/68 on MOSEI, MELD, IEMOCAP-4 and IEMOCAP-6.
Do not rerun seed66. Reuse its11 classification jobs and original MOSEI V3.
Add exactly24 jobs:2MOSEI,2MELD,10IEMOCAP-4 and10IEMOCAP-6. Each IEMOCAP seed
includes all five LOSO folds. Total requested completed suite will contain36runs.

No model/core change. The classification driver accepts two more seeds and
preserves its seed66 behavior; original running jobs use their old immutable
snapshot. MOSEI supplements reuse the tested V3 training loop with an explicit
same-seed setup factory and completed1e-3 Flat references. Seeds genuinely affect
model initialization, training order, mask RNG and logged configuration; all
use the same accepted seed66 data split/evaluation masks for paired comparison.
MELD masks are accepted baseline exact masks; IEMOCAP-6 corrected fixed masks;
IEMOCAP-4 has an explicitly regenerated fixed legal GCNet bank.

| Task | epochs | LR | Python |
|---|---:|---:|---|
| MOSEI |60|1e-3|s0|
| MELD |50|1e-5|cmerc-repro|
| IEMOCAP-4/6 |100|1e-3|cmerc-repro|

All use batch32,Adam,WD1e-5, original MSE/CE,GCNet random-mask-v1, and original
cyclic training. Classification chooses per-rate TEST ACC with same-call UA/WF1;
MOSEI chooses per-rate TEST W-F1 with same-call ACC/MSE. These are internal
oracle exploration, not independent generalization or one checkpoint deployment.

## Persistent execution

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_v3_three_seed_20261021.run \
  --source-commit SOURCE_SHA \
  --output /data2/yb/remote_experiments/osram_v3_three_seed_20261021/execution_v1
```

This creates the24-job ledger immediately, then waits for the existing seed66
classification controller to finish. It does NOT wait for manual user input.
This dependency is necessary because that immutable controller does not count
another controller's processes. When it exits, the new controller automatically
fills physicalGPU0 cap4 andGPU2 cap8 subject to measured per-dataset memory,
hostRAM64GiB and disk20GiB reserves. Other tasks are never killed or moved.
All existing successful GPU smokes are reused; each new worker retains original
zero-init, real-gradient and checkpoint integrity gates. No extra methods/HPO.

QUEUE.json distinguishes waiting/pending/running/completed/failed. A failed
predecessor stops handoff for inspection, not a fake completion. Same-source
worker full checkpoint resume remains explicit; completed outputs cannot be
overwritten. All BEST/same-call predictions/Mask/full last state remain remote.
REPORT.json/PER_RATE.csv/RESULT.md update automatically. Only fully completed
fivefold seeds enter seed summaries and only all3 seeds enter3-seed means.

The queue is a launch/scheduling deliverable. A queued job is not running and
a successful implementation is not evidence that V3 improves performance.
