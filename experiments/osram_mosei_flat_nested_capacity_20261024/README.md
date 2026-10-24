# MOSEI: small Flat and existing Nested capacity

INTERNAL TEST-ORACLE SCREEN. User authorized physical GPU0/1 for more Flat
widths and several Flat+Nested arms. Exactly seven new workers plus the
already-running pure Flat256; never duplicate that worker. No MOSI/seeds/HPO.

| GPU | Arm | Flat hidden | Nested dim / depth / groups / MLP |
|---:|---|---:|---|
| 0 | f128 | 128 | none |
| 0 | f384 | 384 | none |
| 0 | f768 | 768 | none |
| 0 | f256_n64 | 256 | 64 / 3 / 8 / 64, original Nested |
| 1 | f256, observe existing | 256 | none |
| 1 | f256_n128 | 256 | 128 / 3 / 8 / 128 |
| 1 | f512_n128 | 512 | 128 / 3 / 8 / 128 |
| 1 | f512_n256 | 512 | 256 / 3 / 8 / 256 |

All seed66,60epochs, s0 Torch2.2.2+cu121/NumPy1.24.3, constantLR1e-3,
Adam/WD1e-5,batch32,MSE,gcnet-random-mask-v1 cyclic0.0-0.7. Original CHM
8x64/joint BlockWrite/read-before-write/write_step0.6,LocalSkip and output1600
remain unchanged. Nested uses its existing input-correction residual and zero
decoders, not direct replacement. Only hidden width and module switch change.

Reuse the7ec7878 numerical snapshot; the sole numerical-package extension is
one registry entry `mosei_capacity_nested256: {'dim':256}`. Remove that exact
entry and the entire registry must equal the original bytes. No changes to
OSRAM/model/trainer/sampler math; source/data/mask/config/environment hashes
are retained. Existing immutable sources are not edited. The same-width plain
initialization must equal the non-Nested parameter digest. A first-batch eval
checks zero-decoder predictions; first formal epoch checks nonzero finite
Nested gradients and one CHM scan per batch, original mask/update/LR traces.
No separate seven-job GPU smoke sweep, and no extra training loss.

GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`;
GPU1 UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`.
At most4workers per GPU, INCLUDING existing Flat256 on GPU1. Admission charges
actual allocations plus6GiB reserved per worker (8GiB for Nested256),2GiB
margin,64GiB RAM/20GiB disk minimum. Do not change batch/precision to fit; wait
if resources change. GPU4 forbidden, unrelated tasks untouched.

Run from the immutable composite source using a persistent tmux session:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosei_flat_nested_capacity_20261024.run \
 --source-commit RUNNER_SHA --output NEW_INDEPENDENT_OUTPUT
```

Parent source directory contains SOURCE_MANIFEST.json. The queue saves
QUEUE.json and independent logs/full recovery states/eight selected checkpoints
with exact same-call predictions and masks. Explicit --resume requires dead
previous workers and compatible full states; failed arms are not silently
retried. No process is signalled by this controller.

Reports preserve partial states but score only completed runs. Final output
contains eight approved arms plus three read-only plain controls512/1024/1600,
per-rate ACC/W-F1/MSE, high0.5/0.6/0.7 and Nested deltas against SAME-WIDTH
plain Flat (not only original1600). Eight rates can select different TEST-best
epochs. This is an internal one-seed screen, not independent generalization.
Do not mix in cmerc/LR1e-4 or later CUDA RNG-repaired scores.
