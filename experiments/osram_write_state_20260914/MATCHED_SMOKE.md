# Matched two-update stability diagnostic

INTERNAL DIAGNOSTIC ONLY. No full training, architecture changes or F1 evaluation.

## Setup

Seed 66, first real MOSI training batch: 32 conversations, 843 valid utterances,
rate 0.5, 1,142 missing targets. Causal Flat OSRAM, write step 0.6.
Adam LR 0.001, original weight decay, gradient clipping 1.0, dropout 0.5.
Both models start from scratch. All 144 common state tensors are exactly equal.
CPU/CUDA RNG states are restored before each model construction and each paired
forward, so shared dropout paths start from the same random state. Each model
receives two updates on the same prepared batch; each updates its own EMA.

The inherited `config` in JSON is the setup template (emotion-only); the actual
objectives explicitly executed by the script are **Flat Joint JEPA** and
**WSC write-token supervision**, both weighted by 0.1. These are not identical
auxiliary losses; compare emotion loss separately, not just total loss.

## Losses before each optimizer update

| Model | Step | Emotion | Auxiliary (raw) | Auxiliary (weighted) | Total |
|---|---:|---:|---:|---:|---:|
| Flat Joint | 1 | 2.844248 | 3.635936 | 0.363594 | 3.207841 |
| WSC | 1 | 2.844248 | 0.104586 | 0.010459 | 2.854706 |
| Flat Joint | 2 | 40.241238 | 4.352195 | 0.435220 | 40.676456 |
| WSC | 2 | 40.325310 | 0.094740 | 0.009474 | 40.334785 |

## Gradient and actual memory norms

| Model | Step | Total gradient L2 before clip | After clip | Emotion→predictor L2 | Postwrite memory mean | Postwrite memory max |
|---|---:|---:|---:|---:|---:|---:|
| Flat Joint | 1 | 26.9320 | 1.0000 | 0 | 6.1719 | 8.0159 |
| WSC | 1 | 26.9675 | 1.0000 | 0 | 6.6375 | 8.5385 |
| Flat Joint | 2 | 364.5891 | 1.0000 | 0 | 5.9094 | 7.7057 |
| WSC | 2 | 373.6712 | 1.0000 | 31.2323 | 6.2636 | 7.9122 |

Memory norms are actual per-head matrix Frobenius norms of returned block-write
states, pooled over batch/head/scan calls (including calls for padded batch rows).
They are not the legacy `memory_frobenius_norm` field, nor a valid-only retention
metric. JSON additionally records encoder/OSRAM/classifier/predictor gradient L2.

## Interpretation

The previously reported WSC 2.85→40.33 total-loss jump is reproduced. However,
the matched Flat baseline has essentially the same emotion-loss jump. This
observation does **not** support attributing the jump specifically to predicted
memory writes. Nor does it prove that either model is stable over a training run.

WSC emotion loss at step 2 exceeds Flat by 0.084072, approximately 0.21%.
The initial zero-initialized emotion adapter explains why memory/predictor task
gradients do not appear at step 1; their appearance at step 2 is connectivity
evidence, not performance evidence. The exact cause of the common early update
overshoot has not been isolated by this two-step diagnostic.

The independent architectural concerns remain: predicted and observed slots
share write strength, and predicted columns change the joint ridge solve.
This diagnostic neither fixes nor establishes the performance harm of those
choices. No gate, write attenuation, new loss or training run was introduced.

## Reproduction and verification

Run from the remote repository root:

```bash
CUDA_VISIBLE_DEVICES=5 /data2/yb/reproduction_envs/s0/bin/python3.10 \
  experiments/osram_write_state_20260914/matched_smoke.py
```

Completed with exit 0; common initialization equality, finite losses and finite
gradients were asserted. Four model-step records were produced. This is a real
batch integration diagnostic, not a rerun of the full unit-test suite.
