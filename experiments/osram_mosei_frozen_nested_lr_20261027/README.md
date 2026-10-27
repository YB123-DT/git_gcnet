# MOSEI frozen Nested: two learning rates only

Approved finite follow-up, seed66 only. Reuse completed default3e-4 original
Nested and protected Flat; train16 new branches (two LR settings ×eight rates).
No architecture HPO, new seeds, new masks, parent retraining or additional loss.

| Item | Fixed setting |
|---|---|
| New branch LR | 1e-4 or1e-3, constant60epochs |
| Parent | Own-rate original MOSEI66 Flat; originally trained at1e-4 |
| Trainable parameters | Original Nested159235 only, dim64/depth3/MLP64/groups8 |
| Optimizer | Adam,batch32,WD1e-5, original clipping/task MSE including neutral |
| Training missing | Original cyclic GCNet-random-mask-v1, independent host/guest |
| Evaluation | Only own-rate exact protected mask; Test-best W-F1, earliest tie |
| Frozen/eval | Encoder,CHM,Query,Flat1600,LocalSkip,Norm,task head |

Initial branch tensors, original parent outputs, test masks and every formal
epoch's sample/mask/optimizer-step trace must match the completed default Nested.
Only learning_rate changes in the effective stage2 configuration. No unfreezing,
new dropout, altered residual initialization, sequential write or auxiliary loss.

Original Signature/NeuMiss weights were deliberately deleted previously. This
experiment verifies the still-retained **Nested** reference checkpoint instead;
the old numerical source name containing Signature does not mean using its
model, features or missing protocol. All original artifacts remain untouched.

Server: ssh biggpu. Runtime: `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Numerical cwd: `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
Immutable overlay contains this package plus reused frozen_nested_gcnet,
frozen_nested_cross and recent_noarea_serial helpers. No editable core overlay.

```sh
PYTHONPATH=IMMUTABLE_OVERLAY:NUMERIC_ROOT \
 /data2/yb/conda_envs/cmerc-repro/bin/python \
 -m experiments.osram_mosei_frozen_nested_lr_20261027.scheduler \
 --root NEW_RUN_ROOT --source-commit ACTUAL_FULL_OVERLAY_COMMIT --per-gpu 4
```

Two short real GPU smokes precede the16 formal jobs; each smoke is two one-batch
updates, not an epoch-budget or performance claim. Healthy physicalGPU0/1/2 only,
UUID bound, max4/card and12global. Resource admission uses actual free VRAM/RAM,
conservative reserves and completed same-architecture full-run measurements.
Insufficient capacity queues; batch/architecture stay unchanged. GPU4 prohibited.

Controller is tmux-persistent. QUEUE.json retains commands,PIDs,UUID,cwd,logs,
statuses and resource snapshots. `--resume` only restores compatible full
block/Adam/RNG/history states. Source/config changes require a distinct run.
Failures stop new admissions, preserve other live jobs and fail visibly.

After all16 finish, the controller automatically recomputes selected predictions,
checks checkpoint/mask hashes, earliest best epochs, formal training traces and
Frozen-parent identity, and writes RESULT.md/JSON/CSV. Report Flat/default3e-4/
new1e-4/new1e-3 overall and high-missing ACC/W-F1 plus per-rate scores/epochs.

INTERNAL TEST-ORACLE exploration. Eight own-rate parent/branch pairs are not one
deployment checkpoint. Extra stage2 budget and single-seed LR selection remain
limitations. No automatic promotion, multi-seed expansion or structure sweep.
