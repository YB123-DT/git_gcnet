# Fixed Text evidence under changed history context

INTERNAL DIAGNOSTIC ONLY. Approved four-condition design; no training/model changes.

Use original cfg84 Flat MOSI seed66 eight per-rate Test-oracle checkpoints.
For every test target choose the nearest eligible historical Text whose deletion
retains another observed modality. Choose one other legally deletable A/V bit
outside that Text's entire utterance, closest in time to the Text; tie-break by
most recent position then A before V. No labels/predictions enter selection.
Skip targets without both conditions. Other history deletion strength exactly1.
All other Text bits, focal whole utterance and target/future inputs stay fixed.
Each eligible target has exactly one pair, not a search over deletion patterns.

A+/A- use original background with Text present/removed; B+/B- additionally
remove the same one A/V bit. Independent causal scans reset Memory each call.
Primary Delta=MSE(without Text)-MSE(with Text); J=Delta_B-Delta_A.
Report signed/absolute J, contribution sign reversals, polarity rescue/harm,
four-condition W-F1 on the same eligible nonneutral samples, per-rate and
availability/deletion-modality/lag strata. Use >0 polarity; exclude gold0 only
for classification, never for planning/MSE. Also report fixed1e-6 contribution
deadband, not a fitted threshold. Equal-rate summaries, not pooled W-F1.
These interventions measure model behavior; neither redundancy nor helpful-to-
harmful transitions automatically establish a malfunction or semantic causality.

Implementation: diagnostic.py pure planning/mask/stat functions; run.py existing
loader/capture wrappers with frozen strict checkpoint loading; tests in
tests/test_history_text_context.py. No edits to model/trainer.

- [x] RED pure mask/selection/contribution tests.
- [x] Implement four masks and validate no empty masks, unchanged current/future.
- [x] GREEN tests; real checkpoint check of baseline parity and same-current Local.
- [x] Commit/push isolated implementation; deploy over sealed prior audit source
      through a separate diagnostic package, not edits to shared running source.
- [x] Start persistent biggpu healthy GPU7 job with independent outputs and status.
- [ ] Preserve all per-target predictions/identities and code/checkpoint hashes;
      collect final report after all eight rates complete.
