# Local-Conditioned Evidence Gate

Verified optional module on cfg84 no-JEPA Flat. User subsequently authorized
the matched three-seed training experiment; verification is not performance.

## Method

Four scalar gates independently scale Base, Gap-A, Gap-T and Gap-V before the
unchanged Flat adapter; the Local skip and Local adapter input remain intact.
The shared MLP conditions on projected Local/evidence (128 dimensions), their
product, absolute difference and L2 norms, evidence type (8 dimensions), and
availability (3 dimensions). Its hidden layer has 128 units and ELU activation.

`g_i = 1 + 0.2 * tanh(logit_i)`; only the final logit layer is zero-initialized.
The original context, not its projection, is multiplied by the scalar gate.
The added module preserves the original initialization RNG and starts at the
original Flat computation. Original Flat, task head and Memory stay trainable.

Base is active at every valid utterance. A Gap is active only when its modality
is missing. Padding and inactive evidence are safely zeroed using `torch.where`.
Inputs use the emotion contexts after readout ablations, never bypassing them.

`R = sum_valid_active((g_i - 1)^2) / count_valid_active`

`loss_train = loss_emotion + 0.001 * R`

Lambda occurs once. This is a mean over active evidence, so padding and observed
modality gaps do not contribute or dilute the penalty. The regularizer is not a
test-label supervision signal and does not guarantee non-degradation.

## Scope

Keep `--osram-readout-fusion flat`; opt in with `--osram-local-evidence-gate`.
The regularization coefficient is `--osram-local-evidence-gate-reg-weight 0.001`.
The shorter alias `--osram-evidence-gate-reg-weight` is also accepted.
No old total-history gate, Post-GRN, JEPA, completion, persistent 50/50 mixing,
new attention, or frozen original components. The default-off path preserves
the original model. No dependency is added.

Diagnostics report each evidence's active-count-weighted gate mean and fraction
near the bounds (`g <= 0.81` or `g >= 1.19`). An inactive type has a null mean,
not an artificial zero mean. Training also records `classification_loss`,
unweighted regularization and its once-weighted contribution; those losses
retain the trainer's batch-mean aggregation. Evaluation task loss stays unchanged.

Code is edited locally; model verification belongs to `ssh biggpu`, original
`s0` Python environment, host GPU0 UUID
`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`. Host GPU4 is prohibited.

Verification snapshot:
`/data1/yb/remote_experiments/osram_local_evidence_gate_verification_20260930/code`.

The existing oracle analysis motivates the hypothesis but cannot establish
learnability from unlabeled inputs. It used a shared history coefficient;
benefits from four separate gates remain an untested extension of that result.

## Authorized training (2026-09-30)

Reuse original completed Flat seeds 66/67/68, without retraining the reference.
Read each original config from the memory-gap ablation reference. Change only
the evidence-gate enable flag and its regularization weight (0.001); gamma is
fixed 0.2. Train from scratch for the verified original 100 epochs, same cyclic
eight-rate mask schedule, features, split, optimizer and batch size.
Keep eight per-rate best checkpoints. This retains the existing **Test-oracle
internal diagnostic** selection protocol, not validation-selected paper results.

Run from the isolated biggpu snapshot:

```sh
/data2/yb/reproduction_workspace/envs/s0/bin/python -u experiments/osram_local_evidence_gate_20260930/run.py --launch --gpus 0 --max-tasks-per-gpu 3
```

The launcher verifies original configurations and feature paths, refuses
overwrites, locks against duplicate launch, and uses host GPU0 exclusively.
Admission requires at least 3000 MiB free with UUID verification and 20 seconds
between starts. Pending seeds wait on GPU0 instead of changing the protocol or
using another card. Logs/configs/provenance/checkpoints live separately under
`/data1/yb/remote_experiments/osram_local_evidence_gate_20260930/runs`.
