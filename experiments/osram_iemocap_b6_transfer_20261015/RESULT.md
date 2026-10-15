# IEMOCAP B6 transfer — running

INTERNAL TEST-ORACLE SCREEN. No completed task scores yet.

## Actual launch

- Server: `ssh biggpu`; tmux: `iemocap_b6_transfer_1015`.
- Numerical source: `785a01242ef1a82b97d5337acbb33ba963df5537`.
- Launcher/worker implementation: `ab4bd2c6799d0075f36a3d42f5d73ad80d8583a5`.
- Remote output: `/home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015`.
- Finite budget: IEMOCAP4/6 × seeds66/67/68 × folds5,1,2,3,4 =30 fold runs.
- At launch verification:4 running,26 pending,0 failed. GPU1 has2 of ours;
  GPU3 has2. GPU1's unrelated23GB task is untouched; memory admission prevents
  filling the requested eight-process upper bound now. GPU4/5 receive no jobs.
- The first six planned jobs are fold5; after these, the queue fills folds1–4.
  Fold5 results remain separate from complete fivefold averages.
- Original100epochs, batch32, Adam.001, weight decay1e-5, dropout.5,
  GCNet-random-mask-v1 and cyclic0–.7 remain unchanged. Only the selected
  B6 rank2/14-step/threshold and B6-only LR multiplier are transferred.

Running first: IEMOCAP4 seeds66/67/68 and IEMOCAP6 seed66, all fold5.
The formal logs confirm epoch1 training and eight-rate testing completed.
Early epoch scores are NOT final performance results.

## Verification

CPU:41 passed,2 optional skips. Real longest-batch GPU smoke: three updates,
finite losses/gradients, actual B6 parameter updates;3333.87MiB peak allocation.
The only numerical-source trainer change is extending two classification
artifact guards to allow IEMOCAP; forward, loss and RNG equivalence checks pass.

One live full recovery state was checked: model/optimizer/progress are saved,
CUDA RNG contains one real device, and eight BEST references exist. Metrics
recomputed from all eight stored selected-epoch artifacts pass checkpoint ACC,
prediction, label, epoch, mask and file-hash checks. This uses no new inference.
Keep all BEST checkpoints and exact selected artifacts; UA must never come
from a fresh randomly masked evaluation attached to an earlier best ACC.

Existing historical Flat baseline did not retain exact selected-epoch UA.
Do not present its fresh final reevaluation UA as a strictly paired best-ACC UA.

## Persistent command

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_code_20261015/run.py \
  --worker /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_code_20261015/worker.py \
  --smoke-summary /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015_preflight/smoke_v2/SUMMARY.json \
  --data-provenance /home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015_preflight/DATA_PROVENANCE.json
```

The controller is already running: do not launch a second copy. State/logs:
`queue/QUEUE.json`, `queue/logs/`, and `RUNNER.json` under remote output.
The queue automatically refreshes `SUMMARY.json`, `SUMMARY.csv`, `FOLD5.json`
and `RESULT.md` on completion. A successful start/smoke is not performance
evidence; no additional search or dataset is authorized by this launch.
