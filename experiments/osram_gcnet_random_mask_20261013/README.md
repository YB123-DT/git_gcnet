# Original GCNet random-mask rerun

This experiment changes only the OSRAM missing-modality sampler to the
batch-level `random_mask` algorithm in the original GCNet implementation.
ComP code and its existing results are not changed or rerun.

- IEMOCAP-4/6: Flat, seeds 66/67/68, all five folds, 100 epochs.
- MELD: Flat and selected Nested GNN, seeds 66/67/68, 50 epochs.
- selection remains the established per-rate Test-oracle internal protocol.
- Host and Guest follow GCNet and are sampled independently.
- rate 0.7 retains exactly one modality per utterance.

GPU assignment on `biggpu`: IEMOCAP-4 on GPU0, IEMOCAP-6 on GPU1, and
all MELD runs on GPU3. Physical GPU4 is never admitted.
