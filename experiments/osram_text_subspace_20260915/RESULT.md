# Missing-Text predictable subspace: implementation and bounded smoke

Branch: `feature/osram-complete`; implementation base: `423b7b9`.

**Implementation verified; scientific effectiveness NOT established. No full five-seed training was started.**

Follow-up [no-update dynamics audit](DYNAMICS.md): Teacher Text validation erank is already1.644 and initial random R0 gives1.462, versus selected R1.841. The low rank is **not evidence of collapse induced by the two Stage1 updates**. The audit also measures all four R gradient terms and actual weighted Stage2 shared gradients.

## Approved gradient correction

```python
target = R(z_text_teacher.detach())
loss_pred = smooth_l1(Q(observed), target)
```

Teacher is frozen. Sentiment MSE and predictability SmoothL1 both update R. The target may move toward what Audio/Visual can predict; that is intentional. Variance/covariance are safeguards, not proof of rank 32 or task usefulness.

## Changed files and compatibility

- `gcnet_missing_m3/text_subspace.py`: Stage-1 R/C/Q, safeguards/statistics, strict projector checkpoint import, Text-only Stage-2 loss.
- `gcnet_missing_m3/model.py`: optional frozen R after unchanged shared initialization, integrity hash, eval mode. No edits to OSRAM/readout/MMoE forward.
- `gcnet_missing_m3/train_gcnet.py`: appended target-space config/CLI, loss dispatch, actual Text-only target counts, checkpoint provenance/hash checks.
- `experiments/osram_text_subspace_20260915/{run,smoke}.py`: explicit single-seed stage commands and hard-bounded smoke.
- `tests/test_text_subspace{,_integration,_runner}.py`: gradient, freeze, scope, compatibility, selection and smoke-limit regression tests.

No refactor, dependencies, feedback, loss sweep, or backbone changes. Default `target_space=all-modalities` creates no new model weights; old state dictionaries load strictly. Both default and new modes retain original inference, shared initialization and RNG. Stage 2 adds **8,224 frozen parameters**, zero new trainable parameters, zero auxiliary inference calls.

## Stage 1

Only selected supervised online Teacher projectors are instantiated (no OSRAM, no Teacher classifier). Full training feature blocks enter these frozen projectors. Q receives only current Audio and/or Visual latents. R receives detached Text latent. A/V/AV use fixed slots and two availability bits; three prediction losses are averaged.

| Module | Shape | Parameters |
|---|---|---:|
| R | Linear 256→32 | 8,224 |
| Q | Linear 514→128, GELU, Linear 128→32 | 70,048 |
| C | Linear 32→1 | 33 |
| Total trainable in Stage 1 | | 78,305 |

Fixed regularization coefficient beta=1, variance floor=1, covariance squared off-diagonal sum/dimension. These are documented first-version defaults, not tuned values. MOSI source LR=0.001, weight decay=1e-5. Full runner, if later authorized, uses 100 epochs and minimum validation composite loss. The saved selected R/C/Q state supports transfer and diagnostic reproduction, **not optimizer-resume**.

### Real MOSI smoke

Seed 66; GPU 6 UUID `GPU-e4cafb17-818e-216a-b94a-7440063a9153`; one Stage-1 epoch capped at two optimizer updates. Teacher export comes from complete-view supervised epoch 37, selected using validation. Train=1,284 utterances; validation=229. No Test batch was consumed. Stage-1 and Teacher selection share the validation set; these metrics are not an independent generalization estimate.

| Selected Stage-1 measurement | Train | Validation |
|---|---:|---:|
| Sentiment MSE | 1.55696 | 1.97857 |
| Sentiment W-F1, excluding y=0 | 83.859% | 84.386% |
| Predictability SmoothL1, mean A/V/AV | 0.08276 | 0.08490 |
| Variance penalty | 0.70637 | 0.70231 |
| Covariance penalty | 0.33278 | 0.34838 |
| R output effective rank /32 | 1.853 | 1.841 |
| Mean per-dimension std | 0.29339 | 0.29745 |
| Per-dimension std min–max | 0.07669–0.75165 | 0.07087–0.74447 |

**Warning: R is nonconstant but markedly low-rank. High sentiment W-F1 does not establish a diverse predictable subspace.** The JSON's `collapsed` flag only detects near-zero variance or rank<1.5; `false` does not mean representation quality passed. With two updates, neither long-run collapse nor noncollapse can be established.

| Validation prediction | A→Text | V→Text | AV→Text |
|---|---:|---:|---:|
| SmoothL1 in R space | 0.08524 | 0.08351 | 0.08594 |
| Centered cosine | 0.00759 | 0.00730 | 0.01270 |
| Real-minus-shuffle cosine | -0.00014 | -0.00139 | -0.00025 |
| Centered retrieval | 1.747% | 0.873% | 0.873% |
| Chance | 0.437% | 0.437% | 0.437% |
| Prediction/target std ratio | 0.0382 | 0.0952 | 0.1163 |

Predictions are not yet demonstrably sample-specific; the raw cosine near0.58 is not treated as evidence. Full per-dimension std, prediction ranks, covariance and split metrics are in `SMOKE.json`.

First-step gradient L2:

- R from sentiment: **22.5791**.
- R from predictability: **1.09185**.
- Q from predictability: **0.361432**.
- Frozen Teacher: no gradient; projector hash unchanged.

## Stage 2

Fresh baseline Student + original contextual MMoE → original 256d reg/cl outputs → frozen R → missing-Text-only loss. Teacher and R are frozen; Q/C are not imported. Teacher hash must match the Stage-1 artifact. No Audio/Visual target loss. Empty Text target batches safely return differentiable zero. For a singleton Text target, NCE=0 and total=.5 regression, retaining the specified fixed weights.

One real missing=.5 training batch, **one optimizer update**, 380 missing Text targets:

| Quantity | Value |
|---|---:|
| Emotion MSE | 2.64912 |
| Subspace regression | 0.12659 |
| Subspace InfoNCE | 10.85580 |
| JEPA (.5/.5) | 5.49119 |
| Total (emotion + .1 JEPA) | 3.19824 |
| JEPA gradient L2 to reg output | 0.002645 |
| JEPA gradient L2 to cl output | 3.57306 |
| JEPA gradient L2 to predictor/MMoE | 44.0144 |
| JEPA gradient L2 to Student encoder | 24.8536 |
| JEPA gradient L2 to OSRAM | 20.1090 |

Losses and gradients finite; Teacher and R have no gradients and identical hashes before/after update. Audio/Visual prediction-output JEPA gradients are exactly zero. Inference succeeded with R, Teacher and MMoE patched to raise on invocation. No Stage-2 performance evaluation or F1 claim.

## Verification

- **397 tests passed**, one pre-existing PyG deprecation warning.
- Includes new unit/integration/runner tests and existing Missing-M3, OSRAM, B2, write-step, retention, complete/future/write-state, Teacher and per-rate selection tests.
- Initial broad run had one fixture error because the remote checkout lacks `.git`; rerun supplied exact historical `osram.py` via the existing `OSRAM_HISTORICAL_SOURCE_B64` and `OSRAM_B2_HISTORICAL_SOURCE_B64` hooks. No production code changed to bypass the test.
- `git diff --check` and Python compile checks passed.
- Spec review and second-pass correctness review found no remaining critical/important issue.
- Hard-bounded smoke rerun reproduced all losses/gradients above.

## Usage (commands prepared, full training NOT executed)

Remote interpreter: `/data2/yb/reproduction_envs/s0/bin/python3.10`; remote repository `/data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple`.

```bash
# Bounded verification only; output directory must be new.
python experiments/osram_text_subspace_20260915/smoke.py --output /path/to/new_smoke

# Future authorized Stage1; train/validation only, one seed.
python experiments/osram_text_subspace_20260915/run.py --stage stage1 --seed 66 --output /path/to/new_stage1

# Future authorized Stage2; refuses a smoke-only subspace checkpoint.
python experiments/osram_text_subspace_20260915/run.py --stage stage2 --seed 66 --target-space predictable-subspace --subspace-checkpoint /path/to/new_stage1/subspace.pt --output /path/to/new_stage2

# Matched Text-only full-256d target control, no R.
python experiments/osram_text_subspace_20260915/run.py --stage stage2 --seed 66 --target-space full-text --output /path/to/new_full_text
```

Later Stage-2 protocol is **each seed × each rate independently selects its highest Test W-F1 epoch**. No eight-rate mean selection. Such results are INTERNAL DIAGNOSTIC ONLY, not formal paper results. Stage 1 never uses this Test selection rule.

## Remaining questions / stop state

No fully trained R yet. No evidence this target resolves predictable sentiment transfer or improves Student F1. Low effective rank and weak centered prediction quality are explicit unresolved risks, not reasons to add mechanisms automatically. No parameter/temperature/weight searches, no five-seed jobs. Implementation stops here pending user confirmation.
