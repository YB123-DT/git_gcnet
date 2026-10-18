# Seed66 paired task-only: running, not final results

INTERNAL TEST-ORACLE SCREEN. Source driver `654f67f18a9cfb4297080e8b9eb9a7dbb35c74bf`.
User authorized one seed on MELD and IEMOCAP6, not a new multi-seed sweep.

Actual server: `ssh biggpu`; runtime `/data2/yb/conda_envs/cmerc-repro/bin/python`.
All six training jobs are running: MELD1split + IEMOCAP6 fivefold, all seed66.
MELD preserves lr1e-5/wd1e-5/50epochs; IEMOCAP6 lr1e-3/wd1e-5/100epochs.
No original model/default config file was changed. Dedicated outputs and retained
eight BEST/exact selected artifacts/full last_training state for every run.

At the recorded launch snapshot, MELD was5/50; IEMOCAP6 folds1/2/3/4 were1/100,
fold5 was3/100. These are progress, NOT completed experiment scores.
Physical GPU0 hosts IEMOCAP6 folds2/4/5, GPU3 hosts MELD and IEMOCAP6 folds1/3.
Every process is UUID-bound to logical cuda:0. No GPU4, no external job signalled.
All six jobs have now committed a real first epoch; the first epoch checks finite
CE gradients, exact equal-weight task loss, and no gradient in the inherited inert
projector. First integration peak: MELD890.81MiB, IEMOCAP6 fold5 3603.67MiB.

MELD backbone initialization matches the accepted baseline exactly after excluding
only the inert training projector. Its View1 train/selection masks match the
accepted original stream at every committed epoch so far. IEMOCAP uses its
unchanged original GCNet sampler and isolated second-view RNG; no full baseline
training trace was available for direct whole-run hash comparison. In the first
epoch JSON, `original_mask_trace_equal=false` for IEMOCAP means NOT CHECKED, not
an observed mismatch. Likewise initialization `matched_to_accepted_baseline=false`
means that no retained baseline initialization digest was available, not failure
of an equality test. The original isolated-initialization/RNG tests passed in
both numerical sources.

Training semantics remain the old A/rho0.5 experiment: both entire views receive
CE on every valid utterance. It is not anchor-only supervision. Model inference
is a single original forward; no contrastive head, second trajectory or labels
are used for inference. No persistent-pattern training is mixed in.

## Auditable packaging recovery

The first archive omitted an existing committed orchestration dependency:
`experiments/osram_meld_alternate_features_20261017/worker.py` (compact state
helper), SHA256 `66fc911607d06ebc98b7e7ab407aaf786185586eebcd8cb2ac4ab4e6b46a48d1`.
The first IEMOCAP5 worker stopped at importing that helper BEFORE model training;
the other four IEMOCAP jobs were held by the integration gate and never launched.
No IEMOCAP training state/step existed to discard.

The exact file was restored from the SAME committed `654f67f`, without editing
any numerical/training code, parameter, configuration, manifest identity or mask.
Only the coordinator was interrupted and reconnected; the existing MELD training
PID1196840 was left running and not restarted. The scheduler recovery event and
original failure logs are retained. Failed IEMOCAP entries were explicitly
requeued with the same configuration; IEMOCAP5 uses attempt2, others attempt1.
All subsequently completed the real first-epoch check. Include that dependency
when reconstructing this sparse deployment archive from Git.

## Persistent runner and artifacts

Code: `/home/qinjinghui/remote_experiments/osram_paired_meld_iemocap6_20261018/code_654f67f`.
Outputs: `/home/qinjinghui/remote_experiments/osram_paired_meld_iemocap6_20261018/runs_v1`.
tmux: `paired_transfer_1018_resume`; actual process commands/PIDs, UUIDs, epochs,
helper hashes and recovery event are in `LAUNCH.json`. Future state is available
in remote `queue/QUEUE.json`, each run's `EPOCH_STATUS.json` and `SUMMARY.json`.

```bash
cd /home/qinjinghui/remote_experiments/osram_paired_meld_iemocap6_20261018/code_654f67f
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
 experiments/osram_paired_meld_iemocap6_20261018/run.py \
 --output /home/qinjinghui/remote_experiments/osram_paired_meld_iemocap6_20261018/runs_v1 \
 --commit 654f67f18a9cfb4297080e8b9eb9a7dbb35c74bf --resume
```

This continuation attaches existing owned processes rather than duplicating
them. Once all finite jobs complete, the controller automatically produces
RESULTS.json/RESULT.md with original seed66 Flat comparisons for Random8/high3
and seven whole-conversation observed sets. IEMOCAP means require all five folds.
No conclusion about gains is warranted from this launch/integration evidence.

Tests:16 local tests passed (one known unused torch-sparse import warning);
original paired tests also passed10/10 separately in each pinned numerical
checkout. All six real classification first epochs succeeded on GPU. Formal
50/100epoch training and final persistent evaluations are still running.
