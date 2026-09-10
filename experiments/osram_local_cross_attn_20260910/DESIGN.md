# Local-query cross-attention readout

Approved bounded experiment: only replace final OSRAM emotion fusion. Local is
the query/subject; Base and active target Gap slots supply keys and values.
No memory/JEPA/MMoE/completion/write-step modifications.

- Q: LayerNorm(Local) → Linear(256,512).
- K: concat(LayerNorm(context), evidence-type embedding16) → shared Linear(528,512).
- V: LayerNorm(context) → shared Linear(512,512).
- Four heads ×128, scaled dot-product attention, explicit inactive-slot mask.
- Base always active for valid rows; Gap only if that modality is missing.
- Padding excluded before softmax, inactive tensors zeroed before normalization.
- Softmax aggregation across evidence, concatenated head outputs512.
- Dropout(original) → Linear(512,700), final projection weight/bias zero initialized.
- Output: LayerNorm(local_skip(Local) + context_residual). Padding output zero.
- No attention-weight dropout, extra FFN, extra graph or modality completion.

`osram_readout_fusion=local-cross-attn`; default remains `flat`, old gated path
retained. Old flat readout keys remain registered but unused/frozen in cross mode.
Construction isolates new-module CPU RNG; all shared parameters initialize identically.
This changes several aspects versus gated128; it is not an isolated bottleneck test.

## Parameters

Real MOSI setup (latent256, H8 memory, context512, output700):

| Model | Registered parameters | Trainable parameters |
|---|---:|---:|
| Flat | 7,181,217 | 6,321,057 |
| Gated128 | 7,686,482 | 4,536,214 |
| Cross-attention512 | 8,388,305 | 5,238,037 |

New cross fusion: 1,207,088 parameters. Registered totals include frozen Teacher
and retained unused flat modules; do not mistake those for active trainable capacity.

## Locked experiment

MOSI seeds66–70, 100epochs, cyclic eight rates, forward-only OSRAM, native write
step .6; preserve each seed's existing Flat configuration except readout flag and
checkpoint-selection bookkeeping. Fresh initialization, no warm-start.

Every seed × rate independently saves its best Test W-F1 epoch (earliest tie).
No eight-rate mean selection. Mean scores are descriptive only.
Flat is inherited by extracting each rate's maximum from completed 100-epoch
histories, never from the old mean-selected best.pt. Matching test-mask hashes are
required. No Flat retraining, other variants, LR sweeps or additional loss.

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**

## Verification before launch

27 tests passed: new cross-attention masks/zero-init/gradients/memory invariants,
existing local-gated, per-rate selection lifecycle and OSRAM tests. Only existing
PyG deprecation warning. Real one-batch smoke:32conversations/843utterances, two
updates, finite gradients. First-step QKV gradients zero as expected from zero-init;
second-step QKV/type/output gradients nonzero. Teacher gradients absent.
Smoke loss increased (3.208→19.094); smoke verifies execution, not convergence.
Shared state/RNG, Local/Base/Gap and each memory state are unchanged in regression tests.
Independent read-only integration review found no launch blockers.

Diagnostics: active-only per-type mean attention, residual and subject norms,
residual/subject ratio. Stored selected-checkpoint diagnostics are **last batch**,
not dataset-wide averages.

Remote root: `/data2/yb/remote_experiments/osram_local_cross_attn_20260910` on biggpu.
GPU2:seeds66/67/68; GPU3:seeds69/70. Launcher PID2564961. Per-process PIDs and
status in QUEUE.json, logs seedN.log. Existing s0 environment, no new dependencies.
