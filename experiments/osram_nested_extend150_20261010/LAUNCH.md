# Launch record

INTERNAL DIAGNOSTIC ONLY. biggpu GPU7, three concurrent tmux runs.
Wrapper commit f6e4f68; historical model source ad211c01c5317e5520890ba9acb1ae9baf54cdd1.

| Seed | tmux session | PID |
|---|---|---:|
| 66 | nested150_66 | 2451400 |
| 67 | nested150_67 | 2451478 |
| 68 | nested150_68 | 2451563 |

Each seed copies original100epoch recovery versions to an independent directory,
then restores full state and continues through150epochs. GPU UUID:
GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e.
Remote root: /data1/yb/remote_experiments/osram_nested_extend150_20261010.
Logs: logs/seed_SEED.log. Outputs: runs/seed_SEED.

Initial attempt failed before any copied checkpoint or training because the
historical TrainConfig is frozen. Fixed by dataclasses.replace; verified on the
actual historical class that only epochs changes and original stays at100.
Failed logs are retained as seed_SEED_attempt1_failed.log; initial wrapper retained
as run_initial_failed.py. No original artifacts were deleted or changed.

At 03:53:41 UTC all three replacement processes were live, provenance running,
with data loading started; no post100epoch result was yet available. Per-seed
launch provenance is cached alongside this file. Completion must be verified
from150epoch history and final complete provenance, not tmux existence alone.

Subsequent verification: all three reached epoch103 with finite losses and zero
JEPA loss. The first100 history entries match their original run exactly; all
three original last_training checkpoint hashes remain unchanged. Continuation
is actively training, not merely queued or loading data.
