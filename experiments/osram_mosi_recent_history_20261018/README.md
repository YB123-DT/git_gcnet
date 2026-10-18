# MOSI: short-window observed-history residual

INTERNAL DIAGNOSTIC ONLY — per-rate Test-oracle W-F1 selection.

One jointly trained seed66 run. The referenced Flat150 three-seed mean is
80.747% W-F1; the appropriate same-seed reference is 81.068095% overall and
76.352251% at rates .5/.6/.7. These are not a single deployed checkpoint score.

## Only architecture change

For utterance t, collect actual observed modality latents at
`max(0,t-4) <= s < t` from this conversation: at most12 tokens. Project
`[current Local256; original Base forward512]` into a128-dimensional query.
Read with four32-dimensional attention heads and project the128-dimensional
result into1600 dimensions using a zero-initialized Linear.

`hidden = original_emotion_norm(original_flat_anchor + recent_residual)`

The original adapter, Local Skip and task head remain intact. Only the extra
output projection is zero initialized. No-history/padding residuals are
hard-zero even after its bias changes. Source latents remain differentiable:
this is one-stage joint task training, not a frozen probe. Attention dropout
is0, so the branch does not consume the original model's dropout RNG stream.
No modality/relative-position embeddings, auxiliary loss, Gate, JEPA,
completion, persistent mixing or second CHM scan is added.

The extra branch has370,624 parameters, compared with13,509,793 original
parameters (2.743% extra;13,880,417 total). This is a separate exact short-window
cache: it does not claim to improve CHM's stored contents or restore missing
modalities. It adds an inference-time short-history read and associated cost.

## Preserve the actual 80.7 protocol

Original numerical source: `ad211c01c5317e5520890ba9acb1ae9baf54cdd1`, archived
at `/data2/yb/remote_experiments/osram_new40_gpu0123_20261004/source_ad211c0`.
The experiment attaches registered torch hooks without editing that archive:
capture the actual observed latents and current Flat Local/Base, then add the
residual immediately before the existing emotion_norm. Hooks must be installed
again before strictly loading a branch checkpoint; branch parameters are saved
under `osram.recent_history.*` and enter the original backbone optimizer group.
The disabled attachment does nothing and preserves strict legacy loading.

Runtime: biggpu `/data2/yb/reproduction_workspace/envs/s0/bin/python`,
Torch2.2.2+cu121, NumPy1.24.3, sklearn1.3.2. This historical MOSI reference uses
the original ConversationMaskSchedule with frozen evaluation masks, NOT the
later MELD GCNet-random sampler. Do not silently substitute the MELD environment,
features or missing protocol.

Reuse the original MOSI train/dev/test, features, seed66, batch32, Adam,
weight_decay1e-5, dropout.5, projector_dropout.1, cyclic .0–.7, MSE task loss.
Train100epochs at .001; restore complete model/Adam/RNG/BEST/mask state and
continue50epochs at .0001. No source BEST weights are used for initialization.

The runner saves provenance, stage configs, full recovery state, eight BEST
checkpoints, prediction artifacts and per-rate/overall/high-missing summaries.
The masks must hash identically to the original reference. Smoke success is
implementation evidence, not evidence of a performance improvement.

## Verification

`python -m pytest tests/test_recent_history.py tests/test_recent_history_attachment.py tests/test_mosi_recent_history_runner.py -q`

The real-data first forward also checks zero-init on/off prediction and RNG
parity before training. A separate finite real-data smoke precedes the full run.
Host GPU4 is forbidden. The initial run uses healthy physical GPU3, bound by
UUID; all other ongoing experiments remain untouched.
