# MOSI direct-source weighting, seed66

INTERNAL TEST-ORACLE SCREEN

User-approved variant: the SAME coupled Base–Gap Partner Read network produces
gamma=2*sigmoid(score), independently for4roles×3sources. Its last layer remains
zero-initialized, giving gamma=1 on every active slot. Output per role is
sum_s gamma*source_read. REPLACE the original forward Base/Gap read with this
sum: do NOT add original Base/Gap back, do not Softmax-normalize sources, and
do not add another prediction head. Source support/partner/trajectory/Local/
Flat1600/CHM/query/task loss remain unchanged. New parameter count96,487.

Default mode remains`residual`, preserving old checkpoints and coefficient
calculation. New config is`osram_source_role=True,osram_source_role_mode=direct`.
CLI:`--osram-source-role --osram-source-role-mode direct`.

Initialization reconstructs original reads only within float tolerance; it
must NOT be described as guaranteed bitwise equality. The GPU startup gate
checks metadata/masks exactly, predictions at fixed atol=rtol=1e-5, records
actual maximum error and exact-equality status, and keeps the old residual
mode's bitwise initial prediction check unchanged. No artificial compensating
residual is added to conceal reconstruction error.

Scientific distinction: old output R+.25*tanh(score)*source_sum was already
approximately a per-source weight1+.25*tanh(score). This variant simultaneously
changes that range from.75–1.25 to0–2 and directly reconstructs the reads.
It is not a clean ablation of an otherwise identical residual connection.

Exactly one fresh one-stage MOSI seed66/150epoch run. Original GCNetv1 cyclic
0.0–0.7,batch32,Adam,wd1e-5,dropout,MSE; all original and new parameters train.
LR1e-3 through100,1e-4 through150, unchanged. Original s0 environment and data;
per-rate earliest Test-best W-F1, same-call ACC/masks/predictions. No persistent
views, auxiliary loss, JEPA, completion, scalar prediction Router or HPO.
Existing Flat and completed coupled-residual seed66 references are reused;
initialization,config,exact test banks and every epoch's training-mask hash,
update count and LR are checked. Existing MOSEI runs are not modified.

From an immutable source snapshot on biggpu, in persistent tmux:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosi_source_role_direct_20261020.run \
 --output /data2/yb/remote_experiments/osram_mosi_source_role_direct_20261020/execution_v1 \
 --source-commit SOURCE_SHA --gpus GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
```

One brief real GPU smoke first; then ONLY one formal job. Preserve8BEST plus
matched predictions/masks and full Adam/model/RNG last state. Complete-only
report recomputes selected metrics and compares Flat/residual/direct. Resource
gate retains measured VRAM/RAM/disk margins, never GPU4, batch/precision changes
or peer kills. No extra seeds/datasets/search auto-start. Smoke is not a score.

Preflight selected idle physicalGPU0 (UUID above), leaving the running MOSEI
and other work on GPU7 untouched. Both stay on the original biggpu server and
s0 environment. GPU assignment is not a change in training protocol.
