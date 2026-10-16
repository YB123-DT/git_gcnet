# MELD B3: fixed 4 tokens / 3 exchange rounds

INTERNAL TEST-ORACLE SCREEN — seed66 only.

User follow-up: focus tuning on the existing 4-token / 3-round B3 structure, not keep changing token count and exchange rounds. Original Flat1600, raw Local Skip, CHM, GCNet missing, task head/loss and original global training hyperparameters remain unchanged.

## Finite preregistered search

- Exactly **4 new 50-epoch jobs**, bringing prior 28 runs to **32 total**. No automatic expansion.
- Fixed: B3 bottlenecks=4, exchanges=3.
- Only B3-local dimensions `(64,128,256)`, attention heads `(2,4,8)`, FFN width `(128,256,512)` and LR multiplier `(0.1,0.3,1.0)` may change.
- Main model LR=0.001, WD=0.00001, dropout=0.5, batch32, seed66. Thus B3 LR is 0.0001, 0.0003 or 0.001.
- Four jointly sampled proposals require at least one respective nondefault knob: dimension, heads, FFN, local LR. They are **not** one-factor controlled ablations.
- Reserve every previous full six-parameter tuple, including non-4/3 runs. Never rerun duplicates automatically.
- TPE imports **only completed 4/3 observations**; scores of other token/round structures are excluded because those factors are no longer modeled. Expected initial history: original default 4/3 and capacity01. With two matching observations, `n_startup_trials=2`; this is a small-history TPE batch, not strong statistical optimization evidence. Four proposals are produced jointly with constant-liar pending-trial accounting.
- Keep old Optuna integer distributions for serialized `bottlenecks` and `rounds` so historical trials remain compatible; the feasible-candidate constraint forces values exactly `(4,3)` before each proposal.
- Objective unchanged: `0.6*(paired ACC−Flat ACC)+0.4*(paired UA−Flat UA)` in percentage points. Test-best metrics are exported separately, never substituted for paired metrics.

## Immutable numerical implementation and resources

Numerical worker is the already tested checkout `/home/qinjinghui/remote_experiments/osram_meld_b3_capacity_code_6ce8ca6`, Git `6ce8ca60d507fe81dd37d9ce74623104ee2e7098`. New controller code may have a newer commit but does not alter the numerical implementation.

Reuse the parent's largest real-batch smoke (8 tokens, 4 rounds, dim256, heads8, FFN512; peak about3204MiB). No new GPU smoke or model changes. Dataset SHA, exact common masks, original Flat configuration and immutable worker SHA are checked before scheduling.

Server: `ssh biggpu`; Python `/data2/yb/conda_envs/cmerc-repro/bin/python`. Physical GPU3 UUID `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`, maximum four total training processes. The controller waits for the current capacity-four queue to finish before scheduling; it never signals or stops existing tasks. GPU4 is prohibited.

## Persistent launch

Run from a separately committed controller checkout, not by editing the numerical worker checkout:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_meld_b3_fixed43_hpo_20261016.run
```

Launch via tmux. Output is `/home/qinjinghui/remote_experiments/osram_meld_b3_fixed43_hpo_20261016`. `RUNNER.json` distinguishes waiting/running/complete/failed. `REFERENCE.json` freezes historical configurations and matched observations. `PREFLIGHT.json` records both controller and numerical SHAs, original masks/data and reused smoke. New study lives in its own `CAPACITY.sqlite3`; no earlier studies are modified. `PLANNED_JOBS.json`, `queue/QUEUE.json`, per-job logs/checkpoints, `TRIALS.json`, `RESULT.csv`, `RESULT.md`, and `SUMMARY.json` follow the existing audited framework.

## Checks

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -m unittest discover -s experiments/osram_meld_b3_fixed43_hpo_20261016/tests -v
/data2/yb/conda_envs/cmerc-repro/bin/python -m unittest discover -s experiments/osram_meld_b3_capacity_hpo_20261016/tests -v
```

Tests cover fixed structure, full-tuple deduplication, only-matching reference import, exact numerical worker/smoke reuse, finite four-trial execution, required-knob coverage and resume without extra jobs. Fake scheduler tests exercise real Optuna persistence/feedback but do not claim model performance.
