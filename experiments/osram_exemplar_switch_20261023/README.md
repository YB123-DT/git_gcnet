# Frozen exemplar switch audit

MOSI seed66, both completed150-epoch TRAIN-exemplar variants, eight original
per-rate TEST-best W-F1 checkpoints. INTERNAL TEST-ORACLE DIAGNOSTIC.

No training, new parameters, fresh missing masks, epoch reselection or optimizer.
The checkpoint's original Flat/Memory/Local/head parameters stay frozen.
OFF means **exemplar retrieval disabled**, not Memory disabled or Local-only.

`run.py::frozen_readouts` captures the original Local Skip and Flat Adapter
outputs during normal ON inference. Their original sum gives the OFF anchor;
it is classified through the same emotion_norm and smax_fc in the same full
sequence/batch shape. No subtraction of a reconstructed residual is required.
Primary ON and OFF therefore share one causal Memory scan. An additional
existing-wrapper `enabled=False` forward independently verifies OFF exactly
on all valid targets. It is a check, not a second history used for comparison.

Guards include strict selected checkpoint/epoch/training identity, original
source hashes, data/split/seed/TRAIN-bank identity, exact saved GCNet masks,
target/speaker ordering, all686 test targets per rate, ON saved-prediction
parity, exact OFF disabled-path parity, unchanged weights and Torch RNG, and
absence of parameter gradients. All labels are evaluation targets only.

Metrics: nonzero-label polarity ACC/W-F1, threshold pred>0; all-label MSE.
Corrections/harms compare OFF→ON. Eight-rate/high means average per-rate metrics
at the ORIGINAL ON-selected checkpoints. OFF is never independently selected.
OFF−Flat describes a separately trained main path; ON−OFF is a frozen branch
intervention. Their arithmetic score decomposition is not a causal account of
training, and OFF can be an off-training-distribution readout.

## Reproduce

Use immutable tracked source and the original s0 environment on biggpu.
Healthy physicalGPU7 UUID was verified before launch; biggpu GPU4 is forbidden.
The launcher environment is not modified and existing jobs are not signaled.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -m unittest discover \
  -s tests -p 'test_exemplar_switch.py' -v
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_exemplar_switch_20261023.run \
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
  --source-commit 1df92392a88b2714a38316fcff5f11d2b8604404 \
  --output /fresh/independent/output
```

Original training source `d6f6eb19d9ef68403dcfe23a18a7ef66712a9602` is checked
by dependency hashes; the current actual model implementation is used.
Training weights/banks/masks remain under
`/data2/yb/remote_experiments/osram_exemplar_mosi_20261023/execution_v1`.
New audit outputs, source snapshot and logs are under
`/data2/yb/remote_experiments/osram_exemplar_switch_20261023`.

The run refuses an existing output directory. Full prediction NPZ files remain
remote; only small results/identities are committed. A failure remains failed
and cannot enter a completed ranking. No automated retraining/HPO/seeds follow.
