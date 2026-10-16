# Fixed4/3 follow-up registered for automatic execution

INTERNAL TEST-ORACLE SCREEN. This is a verified waiting controller,not four
already-running new training processes.

- Server: `ssh biggpu`; physical GPU3 only,UUID
  `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`; maximum four total GPU processes.
- Persistent tmux session: `meld_b3_fixed43_gpu3_1016`.
- Controller PID:2204280; source `52886f778be548fbe4f5a33345420ed16ad7f3c1`.
- Controller checkout:
  `/home/qinjinghui/remote_experiments/osram_meld_b3_fixed43_controller_52886f7`.
- Output:
  `/home/qinjinghui/remote_experiments/osram_meld_b3_fixed43_hpo_20261016`.
- Numerical workers remain at tested immutable source
  `6ce8ca60d507fe81dd37d9ce74623104ee2e7098`,in
  `/home/qinjinghui/remote_experiments/osram_meld_b3_capacity_code_6ce8ca6`.
- Python: `/data2/yb/conda_envs/cmerc-repro/bin/python`.

Verified `RUNNER.json` state: `waiting_for_capacity_four`. At registration the
previous capacity jobs were at45/43/45/39 of50epochs. They are not stopped or
modified. When their controller reports final completion,the persistent follow-up
automatically freezes matching observations,proposes four new jobs and schedules
them with UUID/memory/process-count admission checks.

All new models fix exactly4bottleneck tokens and3exchange rounds. Only B3 internal
dimension,attention heads,FFN width and adapter-only learning rate can vary.
Original Flat1600,raw Local Skip,CHM,global optimizer settings and corrected GCNet
missing are unchanged. Models train from scratch,not from the60.757 checkpoint.

Search history contains only completed4/3 observations; other structures are
reserved for duplicate prevention,not imported as fixed-structure evidence.
Two matching observations are expected,so this is a small-history TPE batch,
not a well-established optimized result. Each new proposal must change a different
required local axis; multiple local settings can change together.

Budget:four new jobs,32combined across prior stages. No automatic search expansion.
The completed default4/3 reference is ACC60.756705/UA37.357830,eight-rate Test-best;
new results will be reported separately from paired-mask replay.

Verification:local6tests passed with2Optuna skips; the deployed remote suite had
**8passed,none skipped**,including actual SQLite/TPE fixed4/3 constraints,
only-matching reference import,full-tuple uniqueness,four-job limit and resume.
Compilation and whitespace checks passed. No numerical model changes and no new
GPU smoke were needed; the same-source largest-structure3204.083MiB smoke is reused.

## Automatic handoff verified: 2026-10-16 11:54:39 UTC

The preceding four capacity jobs completed all50epochs. The fixed4/3 controller
then automatically started all four authorized jobs on physical GPU3. Its
`RUNNER.json` now reports `running`; no new workload was submitted for this status
check. Controller and numerical source SHAs remain as recorded above.

All new jobs were running at11/50epochs. Below are eight-rate intermediate
Test-best ACC and corresponding seven-class UA, in percent. Each UA was computed
directly from its rate's saved selected-epoch labels/logits, not from another
random-mask evaluation. Recomputed ACC matched selection metadata within1e-8.
These are not final results and are not common-mask paired replay results.

| Job | PID | B3 dim | B3 heads | FFN | B3 LR | Epochs | ACC | UA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b3_fixed43_01 |2283933|256|4|256|0.001|11/50|60.536398|34.652029|
| b3_fixed43_02 |2284075|128|8|128|0.001|11/50|60.387931|34.957441|
| b3_fixed43_03 |2284175|128|4|128|0.0003|11/50|60.450192|35.225104|
| b3_fixed43_04 |2284199|128|4|256|0.0003|11/50|60.272989|33.018594|

Every row fixes4tokens/3rounds,seed66 and the original global LR0.001. B3 heads
are internal attention heads,not the unchanged eight CHM Memory heads. Only the
already-authorized four local configurations are running; there is no budget
expansion or additional training/inference from this check.
