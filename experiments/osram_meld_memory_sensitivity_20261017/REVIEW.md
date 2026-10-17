# MELD Memory sensitivity: source audit before parameter intervention

INTERNAL DIAGNOSTIC ONLY. Source inspected at accepted baseline commit
`b4a313841f098487a6482d2712bdac94aa8e319e`,not inferred from newer variants.
No model code,weights,data,masks or training settings were modified.

## Actual parameter meanings

| Quantity | Actual implementation | Interpretation boundary |
|---|---|---|
| alpha | sigmoid(alpha_logits),shape[8],shared over time/modalities within a head | Multiplies the previous state before read;not the readout history-scale alpha from prior experiments. |
| beta | sigmoid(beta_logits),shape[8,3],beta_mode=embedded in accepted configuration | Scales observed K/V columns by sqrt(beta) inside the joint ridge solve;not an external write-step multiplier. |
| write_step | fixed scalar0.6 | Multiplies the complete jointly solved block-delta correction. |
| eta | 1-cos(q,q_res)*norm(q_res)/norm(q) | Derived per-example/head addressing diagnostic,not a learned parameter. |

Source locations at b4a3138:osram.py alpha/beta parameters1064/1067,
block_write1421–1486,_scan1523–1583,and diagnostics1987–1996.

For a single observed unit-norm key k,with M denoting the pre-write state:

```
M_new = M + write_step * beta/(beta+ridge) * (v-Mk)k^T
```

At ridge0.001,beta0.4/0.5/0.6/0.8 yields factors
0.997506/0.998004/0.998336/0.998752. Hence a small beta intervention can be
much weaker than an equally sized write_step intervention. Small beta gradients
need not indicate sigmoid saturation or an unused writer. This identity alone
does not establish insensitivity for correlated,multiple observed keys,or the
learned checkpoint beta values. Do not change beta_mode while calling it a
hyperparameter-only comparison.

Independent float64 NumPy synthetic comparison of the joint-ridge expression
with this identity passed for four beta settings,max discrepancy8.89e-16.
This is algebra verification,not learned-state or performance evidence.

## What the first diagnostic can and cannot recover

- Three cyclic-trained seeds have eight rate-selected BESTs each:24selected
  checkpoint states,not24independently trained models. Report perseed/rate/epoch;
  selected states are distinct from the final epoch and may share model weights.
- Extract actual alpha/beta matrices and sigmoid derivatives,not only logits or
  initialization defaults. Alpha near1does not alone prove excessive retention;
  alpha^lag describes retention alone,not survival through intervening writes.
- Head read/state norms are descriptive. A low norm is not proof of no task use;
  paired leave-head-out task effects are a stronger functional check.
- Ordinary BEST model state_dict files do not preserve historical training
  parameter.grad values. A backward-only replay on fixed training batches can
  measure current task sensitivity (preclip gradients),not recover training-time
  gradients. No optimizer step or parameter update is needed for that replay.

## Scope of subsequent tests (not launched)

Before any scan:strict source/config restoration and normal-prediction parity on
the exact ACC-selected masks. Keep original ECERC features,Local Skip,Query,
causal read-before-write,and single joint A/T/V Block Write. Change one admitted
Memory parameter family per condition. Rescan from zero independently for each
condition;do not reuse original states after changing retention/write settings.

Report same-row CE/ACC/UA/W-F1,corrections/harms and true-versus-fixed-invalid
history advantage. Use the same label-free matched donor definition and coverage
throughout. A task-score gain alone need not establish better use of history.
Do not treat eta as an adjustable parameter or source-key reconstruction gains
as proof of improved emotion decisions. Existing write/retention and Query/
Decision audits remain evidence with their original protocol boundaries.

Only consider a small from-scratch write_step0.4/0.6/0.8 comparison after the
frozen audit is informative. This is a proposal,not training authorization.
Latest user instruction specifies validation selection for final hyperparameters;
existing per-rate Test-oracle diagnostics remain labeled and are not relabeled.

## Historical execution boundary (resolved for parameter extraction)

Three relevant SSH attempts this turn timed out during banner exchange on
`ssh biggpu`,before checkpoint extraction. Thus no new actual alpha/beta values,
head-usage findings or gradients were obtained. A local scan found no archived
MELD JSON containing memory_alpha/memory_beta. The numerical source was available
locally through git show,so the source/algebra audit above is complete.

No remote job was submitted,no GPU allocated,no process signalled,and no local
training substituted. Remote weights must be read once connectivity returns;
do not report this source audit as completion of the proposed three-stage study.

## Subsequent read-only parameter extraction

SSH recovered on2026-10-17. Actual alpha/beta were extracted from all24accepted
MELD Flat BEST checkpoint states. All file hashes/epochs/ACCs match the accepted
baseline archives. See RESULT.md,ALPHA_BETA.json and ALPHA_BETA_PER_HEAD.csv.
This completes the static parameter check only;there was no training,backward,
head-use evaluation or frozen parameter intervention. Earlier connectivity
failure is retained above as historical evidence,not the current extraction status.
