# GCNet Unified Experiment Protocol Design

## Goal

Build one leakage-free, paired, reproducible experiment protocol for
IEMOCAP-Four, IEMOCAP-Six, CMU-MOSI, and CMU-MOSEI. Baseline and JEPA must see
the same shared initialization, sample order, missing masks, validation rule,
and test examples. Every reported result must be traceable to a manifest.

## Alternatives considered

1. **Patch the current monolithic trainer.** Smallest diff, but random sampling,
   splitting, training, evaluation, and reporting remain coupled. This is easy
   to break when a new model adds a forward pass.
2. **Add focused protocol modules and keep the GCNet model/training loop.** This
   is the selected approach. It limits model risk while isolating each source of
   experimental variance behind a tested interface.
3. **Rewrite the complete trainer around a new framework.** Architecturally
   clean but too risky while reproducing a legacy TPAMI implementation.

## Architecture

### Seed isolation and sample order

A `SeedBundle` derives independent seeds for model initialization, data order,
missing masks, and training stochasticity. An epoch-aware sampler owns a local
`torch.Generator`; dropout or an extra JEPA forward can no longer change the
next epoch's sample order.

### Deterministic mask schedule

Missing masks are generated per conversation from a stable hash of dataset,
split, fold, epoch, conversation ID, side, rate, and mask seed. Training masks
may vary deterministically by epoch. Validation and test always use epoch zero
and therefore remain fixed. Baseline and JEPA manifests must report the same
mask-schedule hash for a paired condition.

The schedule records both requested and realized missing rates. With three
modalities and the at-least-one-view constraint, requested 0.7 is reported with
its actual realized rate instead of being described as exactly 70%.

### Leakage-free splits

MOSI and MOSEI retain their official train/validation/test video sets.
IEMOCAP uses leave-one-session-out test folds. Validation conversations are
selected only from the four non-test sessions by a deterministic,
conversation-level stratified allocation. No test conversation participates in
normalization, early stopping, mask tuning, or model selection.

### Shared initialization

Shared GCNet encoder and regression/classification-head tensors are identified
by name and hashed. A paired Baseline/JEPA run either loads one shared
checkpoint or proves an identical shared-state hash before training. Variant
heads are excluded from this hash.

### Common stability objective

Both add-on GCNet and clean replacement JEPA expose the same training-only
stability decoder. At complete-modality training, a deterministic masked
auxiliary view reconstructs missing features; validation, test, and deployment
use only the main path. The stability decoder and weight are identical across
paired variants.

The original GCNet reconstruction objective and the JEPA objective remain
method-specific tasks. They are not used as unequal hidden stabilizers.

### Missing-loss normalization

Reconstruction losses are divided by the number of missing, real feature
elements rather than all utterances. Their effective scale therefore does not
grow automatically with the missing rate. Empty selections produce a finite,
graph-connected zero.

### Validation and test lifecycle

Each epoch runs train and validation only. Validation weighted F1 selects the
checkpoint. Test is evaluated exactly once after restoring that checkpoint.
Test metrics and predictions are never consulted during training.

### Environment and manifest

The canonical formal environment is the checked-in `gcnet-official`
environment. Each run records Python, PyTorch, CUDA, cuDNN, PyG, NumPy,
scikit-learn, GPU model/index, driver, command, git revision, feature paths and
hashes, split hash, seed bundle, sampler hash, mask hash, shared-init hash,
requested/realized missing rates, and checkpoint-selection metric.

CUDA `scatter_add` is known to lack a deterministic implementation. Formal
claims therefore use paired repetitions, report mean, standard deviation,
paired deltas, and collapse counts, rather than claiming bitwise determinism.

## Error handling

- Reject overlapping train/validation/test conversation IDs.
- Reject a test mask schedule that changes across epochs.
- Reject paired manifests with different split, mask, sampler, feature, or
  shared-initialization hashes.
- Reject a stability objective enabled for only one member of a paired run.
- Fail on empty training/validation/test splits, unknown sessions, NaN losses,
  or missing feature files.

## Verification

Unit tests must prove RNG isolation, epoch-specific sampler repeatability,
conversation-keyed masks, fixed evaluation masks, realized-rate reporting,
split disjointness, shared-state parity, missing-count normalization, common
stabilizer parity, and single test evaluation.

Smoke tests run one short fold/job for all four datasets and both variants in
the canonical environment. A paired parity smoke must show identical shared
initialization, sample-order hash, and mask hash.

## Success criteria

1. No IEMOCAP test conversation appears in train or validation.
2. Baseline and JEPA paired manifests have identical data/split/mask/order/init
   hashes.
3. Test is invoked once per fold after checkpoint restoration.
4. Missing reconstruction scale is invariant to duplicating the count of equal
   missing errors.
5. The common stability path works for both add-on and replacement variants and
   is absent during evaluation.
6. All unit tests and four-dataset smoke tests pass without NaN or leakage.
