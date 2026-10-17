# Restore the retained OSRAM dynamic missingness protocol

User override on 2026-10-17: use retained OSRAM Flat gcnet-random-mask-v1, superseding fixed-test-mask requirement.

1. Reuse OSRAM legacy sampler byte-for-byte including MT19937, host/guest padded rectangle and OSRAM B/T mapping. No model architecture, padding loss, graph or attention changes.
2. Dynamic train/test; save selected epoch predictions/masks immediately; replay its saved evaluation RNG for verification. Preserve Test-best ACC, earliest ties and Stage2-only selection.
3. Isolate code/configs/runs; preserve fixed-mask artifacts, reuse compatible full-modality MMIN teachers, accepted historical ComP and six all-visible Session5 runs.
4. Verify exact sampler/RNG, test-mask changes, checkpoint/mask replay, metrics and unchanged architecture source hashes.
5. Real GPU smoke gates before formal tasks. Scope only IEMOCAP4/6 Session5 three seeds/eight MR/main baselines. No ComP retraining, other folds or appendix.
6. Withdraw fixed-mask upload draft; publish only accurately labeled actual results/status.
