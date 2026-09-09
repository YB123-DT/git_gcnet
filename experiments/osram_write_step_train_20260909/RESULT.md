# Fixed-step retraining — RUNNING

INTERNAL DIAGNOSTIC ONLY. No retraining result is available yet.

IEMOCAP-4 seeds66–70 launched from scratch on2026-09-09 UTC. All five passed
initial training epochs. GPU0: seeds66/67/68; GPU1: seeds69/70.
Coordinator PID2052794. MOSI is queued behind all five IEMOCAP training and
C/D evaluations, not running yet at this snapshot. LAUNCH.json records exact
configuration deltas, original checkpoint hashes and source-file hashes.

Only osram_write_step changes from1 to.6; optimizer learning_rate=.001.
All originally trainable modules remain trainable. No source checkpoint weights
are loaded into the new model. Each worker merely reads the old checkpoint's
metadata to audit provenance, then discards it before run_experiment.

Training uses the original cyclic eight-rate schedule and100epoch budget.
Best checkpoint uses the inherited eight-rate-mean Test-oracle rule.
Cross evaluation uses that same new checkpoint twice at native step.6 and1;
no extra training or checkpoint reselection. A/B/C/D paired tables will use
five rates0/.1/.3/.5/.7; normal eight-rate training results remain separate.

Implemented: differentiable normal block_write, config/CLI/default1 compatibility,
explicit evaluation-only step override, and sequential-dataset concurrent-seed runner.
The frozen inference intervention is not used for training. Nonunit native
steps cannot silently combine with legacy fixed-step intervention multipliers.

Verified: 23 OSRAM tests (including historical forward/backward/RNG exact match,
native.6 vs actual frozen intervention, causal/padding, config restore);
36 existing CLI/config regressions; 3 runner/evaluation tests. No training smoke.
Cross-summary tests:8 passed; absent real C/D inputs correctly return PENDING
without creating a result table. Full paired aggregation remains pending.
The only warning is an existing PyG deprecation. git diff --check passed.

Remote progress:
/data2/yb/remote_experiments/osram_write_step_train_20260909/QUEUE.json

Logs:
/data2/yb/remote_experiments/osram_write_step_train_20260909/iemocap4_seed66.log
(analogous seed67–70 and later mosi_seed66–70).

Do not interpret previous frozen curve numbers as this retraining experiment.
Eta.6 was chosen after seeing Test results; this remains exploratory even if
retraining improves scores. Do not launch another eta without a new task.

After C/D files are complete, aggregate directly on biggpu with:

```bash
python experiments/osram_write_step_train_20260909/summarize.py --remote-root /data2/yb/remote_experiments
```
