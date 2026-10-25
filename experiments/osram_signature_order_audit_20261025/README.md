# Signature mechanism audit: bounded motivation check

Question: does the learned Signature correction use strictly-past path structure
that is additional to the frozen Local/Base/Gap inputs? CHM itself is recurrent
and order-sensitive; an order-blind Flat is NOT the premise.

Use only completed MOSI seed66 cyclic Signature BEST checkpoints at the eight
original rates. Each branch keeps its own original frozen parent and saved mask.
No new weights, training, hyperparameter search, selection or data changes.

## Predeclared controls

1. `flat`: original parent, no extra branch.
2. `normal`: unchanged Signature; must replay selected predictions and polarity.
3. `interior_shuffle_6601/6602/6603`: shuffle observed projected latent values
   only among interior points of each modal window, retaining observed endpoints,
   multiset, count, time coordinates and masks. Local/Base/Gap stay fixed. Only
   paths with at least four observed points permit a nonidentity interior shuffle.
   An identity draw is replaced by an interior swap; this is not a uniform sampler
   over nonidentity permutations.
4. `degree2_no_area`: replace second level with half the outer product of first
   level with itself. Keep all first-level and symmetric second-level information;
   remove only antisymmetric oriented area, including time/latent area terms.

The first level with the actual fixed zero basepoint is an endpoint descriptor,
not a temporal mean. Degree-two signatures are not sufficient to uniquely recover
the history or its arbitrary temporal order. The implementation computes each
modal path separately, not cross-modal joint signatures.

All controls decode through the same frozen original Norm/head. Do NOT rescan
CHM using shuffled inputs; interventions concern only the additional branch.
No-history and padding corrections remain zero. Three shuffle draws are inference
controls, not three training seeds. All prediction outputs go to a new directory.

## Report and decision rule

Report all eight rates, arithmetic eight-rate means, ACC/WF1/MSE, same-weight
corrections/harms, prediction changes, and conversation-cluster bootstrap95%
intervals for paired MSE differences. Also report shuffle-eligible targets,
no-history targets, and current Text availability / prior Text support strata.
Neutral labels remain in MSE and are excluded only from polarity ACC/WF1.

If normal beats Flat and controls consistently lose some of that gain, particularly
on eligible targets, that supports task use of these particular path features.
If controls leave predictions and gain unchanged, do not present ordered-path
information as the verified reason for the improvement. If control removal helps,
report harmful learned dependence rather than claim successful temporal modeling.
Mixed or tiny effects must remain inconclusive; there is no forced positive result.

This is a frozen internal Test-oracle mechanism intervention. Its out-of-distribution
controls cannot establish that all order-free learned alternatives are inferior.
An eventual mechanism claim requires an equal-budget trained comparison to the
same projected observations/timestamps/counts without oriented second-level terms.
That additional training is NOT authorized or started by this audit.

## Reproduce

Run with Python cmerc-repro on biggpu. The numerical import cwd remains the
unchanged `osram_signature_cyclic_multi_20261025/source_e0ca94c`; the new audit
package is an independent namespace-package overlay on PYTHONPATH. Preserve both
the numerical SHA `e0ca94c4c33de0efda2c2f4786fd289067e1ab2c` and the audit commit.

```bash
PYTHONPATH=/path/to/audit_overlay CUDA_VISIBLE_DEVICES=HEALTHY_UUID \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_signature_order_audit_20261025.audit \
 --selected-root /data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/execution_v2/train \
 --output /path/to/new/audit/execution_v1 --source-commit AUDIT_SHA \
 --gpu-uuid HEALTHY_UUID
```

One short GPU inference process, not another training job. Use only the currently
authorized physical0/1/2 UUIDs, never GPU4 or other users' tasks. Shared-resource
runtime is not a controlled speed benchmark. Full predictions stay remote;
commit only small reports, tests, configuration and aggregate statistics.

## IEMOCAP-6 seed68 follow-up

The user selected seed68 for mechanism analysis after seeing its ACC/UA gains.
The independent `iemocap.py` entry point reuses the SAME controls but reports
both ACC and fixed-six-class UA,including label0; no sentiment zero-label filter
or MSE criterion is used. Default scope is all5folds×8rates,40selected branches.
All means keep folds/rates equally weighted. Empty strata remain N/A and their
nonempty-cell coverage is explicit. No new checkpoint selection or training.

Numerical cwd is the unchanged IEMOCAP snapshot `source_21e46f2`,SHA
`21e46f2ecd092345818659694db8ff372672c35e`. Run the new namespace overlay:

```bash
PYTHONPATH=/path/to/new/audit_overlay CUDA_VISIBLE_DEVICES=HEALTHY_UUID \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_signature_order_audit_20261025.iemocap \
 --selected-root /data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/execution_v1/train \
 --output /path/to/new/audit/execution_iemocap68_v1 --source-commit AUDIT_SHA \
 --gpu-uuid HEALTHY_UUID
```

No extra seeds or new trainable models are implied. Positive seed68 sensitivity
would still need to be distinguished from independent multi-seed mechanism
validation; negative evidence is retained along with the measured branch gains.
