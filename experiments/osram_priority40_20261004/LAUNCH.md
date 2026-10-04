# M01–M40 resumed launch

INTERNAL DIAGNOSTIC ONLY

The user resumed training after the code-only delivery at `2ee94eb`.
This supersedes the pause recorded in RESULT.md, not the historical verification evidence.
The batch contains the explicitly requested M01–M40 configurations. Historical mechanism
overlaps remain comparison controls, not additional novel methods. V3's 22 configurations
are not part of this launch. The stopped PointCNN waiter remains stopped.

## Fixed protocol

- MOSI, seed 66, 100 epochs, eight cyclic missing rates 0.0–0.7.
- Exact saved cfg84 baseline; only `osram_meaningful_block` changes.
- No change to task loss, Memory, Query, write, evaluation masks, batch, or optimizer.
- Per-rate BEST selected on Test: exploratory internal screening, **not validation-selected paper results**.
- Independent immutable code snapshot, run directory, configuration, log and checkpoints.
- biggpu physical GPU2/GPU6 only, verified UUIDs; GPU4 forbidden.
- Maximum 11 concurrent jobs per GPU INCLUDING pre-existing and initializing processes.
- No new GPU smoke, explicitly deferred by user. Never record the estimate as a measured peak.
- Actual memory allocation/high-water reserve and free disk constrain admission below that maximum.
- Existing runs are neither stopped nor modified; new candidates wait and automatically fill capacity.

## Evidence and queue

`launch_plan/PART1.json` and `PART2.json` contain exactly 40 ordered configurations.
`launch_cpu_evidence.log`: real project-environment integration check, 3 tests passed
in 17.110 seconds. The earlier full CPU suite is recorded in VERIFICATION.json.
`tests/test_priority40_dispatch.py`: 6 scheduler tests passed; no model training involved.

`dispatch.py` reuses the existing `occupied_lane.py` and training runner. The controller
is copied into a directory without a sibling queue.py to avoid shadowing Python's stdlib.
Two twenty-card manifests use separate runner roots under one forty-card dispatcher.
Only pre-creation resource rejection is retried. Training failures remain recorded failures;
an ambiguous launch PID stops further submission until inspected.

Deployment command, PID and observed state will be recorded in LAUNCH_STATUS.json after
remote startup. Queued is not running; running is not completed. There are no new W-F1
results at this preparation checkpoint.
