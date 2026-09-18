# Modality-local emotion tracks on causal OSRAM

## Objective

This internal diagnostic tests whether replacing the final emotion readout with
explicit modality-local tracks improves the no-JEPA causal OSRAM model. It is
not a change to the conversational memory mechanism.

## Architecture

The control and treatment use the same causal OSRAM backbone, `eta=0.6`,
cyclic mixed-rate training, Student/Teacher-disabled emotion-only objective,
missing masks, and classifier. The only treatment change is the classification
readout:

```text
Observed modality latents + modality embeddings
    -> shared local path, one track per A/T/V
    -> hard-zero missing tracks
    -> concatenate [track_A, track_T, track_V,
                   Base, missing-masked Gap_A/Gap_T/Gap_V,
                   availability]
    -> flat MLP -> emotion hidden
```

The OSRAM fused node, memory scan, Base/Gap reads, queries, keys, values, block
write, read-before-write order, and write step are unchanged. Missing modality
tracks are used only by the emotion readout; they do not alter persistent
memory or any JEPA/MMoE path.

## Protocol

- Dataset: CMU-MOSI
- Seeds: 66, 67, 68, 69, 70
- Missing rates: 0.0--0.7
- Training schedule: cyclic mixed-rate
- Checkpoint selection: independent Test-oracle per seed and per missing rate
- Comparison: existing no-JEPA causal OSRAM control with the same masks
- Status: internal diagnostic only; not a formal paper result

## Verification

The targeted OSRAM/readout test set and a real one-batch forward/backward smoke
test passed. The modality-track tests explicitly verify that the new readout
does not change memory/query/key/value outputs and that inactive tracks are
masked. The full remote test collection contains unrelated historical failures
from the remote checkout (old repository-name, missing historical commit, and
PAM/pretrained-teacher compatibility tests); those are not part of this
diagnostic's acceptance gate.

See [`RESULT.md`](RESULT.md) and `results/` for the complete five-seed tables,
per-rate selections, pattern summaries, and provenance files.
