# Existing paired-view models on the identical Text-history intervention

Frozen inference only, seed66, original saved NPZ masks and anchor identities.
Flat predictions reused; A/B evaluated on biggpu GPU6, PID2611503 (completed).
Remote root `/data1/yb/remote_experiments/osram_paired_text_drift_20261002`.
Launch command and hashes: `paired_results/launch.json`, each arm's provenance/status.
No checkpoint, model, loss, or training change; training-only projector calls=0.

## Findings

- Test flip rates decrease: Flat3.587%, A2.268%, B1.802%. B is more stable by this
  boundary-crossing measure on the exact same anchors. Validation rates also
  decrease (2.294%,1.932%,1.078%). This is one seed, not a significance claim.
- Continuous prediction shift falls mostly with paired task-only training:
  Flat.06894 → A.04955 → B.04941 on test. B offers almost no additional reduction
  in this mean beyond A; on validation B.04874 exceeds A.04646.
- Test anchor W-F1 before/after Text deletion: Flat80.644→80.113,
  A80.309→79.695, B79.884→79.632. B loses less under deletion (-.252pp vs
  A-.614pp), but its resulting absolute W-F1 is still .481pp below Flat and
  .063pp below A. Smaller sensitivity does not recover the accuracy deficit.
- A versus Flat reduces flips but increases deletion-induced W-F1 loss.
  Counts clarify why: Flat52 harmful/40 corrective flips; A37/22; B24/22.
  Suppressing flips can suppress corrections too. W-F1 is nonlinear and rate
  averages cannot be reconstructed simply from pooled flip-count differences.
- Normal full-test W-F1 remains Flat81.068, A80.386, B80.267. These numbers
  use all eligible utterances; the intervention table uses only identical anchors.

Conclusion: this B checkpoint is less likely to change sentiment class when
history Text is removed, but that stability is not accompanied by better W-F1.
Do not attribute all stability to InfoNCE: A already achieves most continuous
prediction-shift reduction. No retraining, tuning, or new module is warranted
as an automatic part of this diagnostic. All scores are internal Test-oracle
checkpoint diagnostics, not formal validation-selected performance.

## Verification

- 32 completed cells: A/B × validation/test × eight rates.
- All 16 normal test W-F1 values reproduce each arm's original reference exactly.
- Same saved masks/anchor IDs/labels; raw current features equal; Local within
  calibrated float32 floor; identical-input repeat and causal-prefix checks pass.
- Model and checkpoint SHA256 unchanged; projector calls0; source artifact and
  output CSV hashes verified in postprocessing.
- 12 local unit tests pass, including new saved-pair rejection tests and metric
  filtering/flip-direction tests; scripts compile successfully.
- Original Flat flip counts/rates reproduce exactly. A validation prediction-shift
  mean differs from the previous calculation by 2.35e-10 because saved float32
  predictions are subtracted as Python float64 during this summary; this does not
  affect predictions, classification, W-F1 or reported precision.
