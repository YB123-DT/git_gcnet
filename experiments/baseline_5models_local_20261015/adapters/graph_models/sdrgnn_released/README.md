# SDR-GNN released-code-compatible adapter

Select exactly `model_version = released_compatible`. The existing
`implementation_fixed` path remains available; omitting the version retains
that existing default. Unknown SDR-GNN versions raise an error.

This package starts from the upstream files identified and hashed in
`PROVENANCE.json`. `adaptation.patch` lists the complete source adaptations:

- Package-relative graph import.
- Replace the hardcoded attention width 2560 with A+T+V and use the largest
  exact divisor no greater than 256 as the head count. MELD uses 1666/238;
  IEMOCAP retains 2560/256.
- Add stable multi-speaker relations for MELD. Temporal IDs are
  past/now/future = 0/1/2; speaker key `ij` maps to `9*i+j`. Original code for
  one or two speakers, including its set iteration and edge order, is retained.
  Freeze `PYTHONHASHSEED` before launching IEMOCAP Python processes to preserve
  reproducible relation numbering. Setting it after interpreter startup is
  insufficient.

The released behavior is retained even where it differs from intended method
semantics. Both graph networks execute and return single-element tuples;
tuple addition followed by `[0]` selects only the temporal graph for
classification and reconstruction. The speaker graph receives no classification
gradient. Reconstruction attention still receives B,T,D with
`batch_first=False`, so its sequence axis is the batch of dialogues. The
official reconstruction loss still flattens T,B,D predictions against a B,T
padding mask. `loss.py` is byte-identical to upstream. The wrapper calls these
official loss classes with the exact train.py transpose for classification.

These are separate algorithms from `implementation_fixed`. Do not merge their
results or transplant a checkpoint across their version directories. MELD is
explicitly a compatibility adaptation; the package is not an assertion that
upstream natively supports MELD.

CPU verification:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python experiments/baseline_5models_local_20261015/adapters/graph_models/sdrgnn_released/test_released.py
PYTHONDONTWRITEBYTECODE=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python experiments/baseline_5models_local_20261015/adapters/graph_models/test_graph_models.py
```

Verified locally: eight released-path tests plus three pre-existing graph tests
pass. The independent local-upstream oracle checks IEMOCAP model initialization,
forward logits/reconstruction/hidden, variable-length loss, and every parameter
gradient with exact zero tolerance. That oracle is skipped when the referenced
local upstream checkout is unavailable; the other tests are self-contained.
