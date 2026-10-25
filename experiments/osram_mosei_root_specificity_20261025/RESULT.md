# Same-weight root-specificity removal — MOSEI N10

INTERNAL DIAGNOSTIC ONLY. Completed frozen inference, not a new trained model.

## Answer

Replacing only each node's rooted-specific input with the existing public
rooted mean causes **zero polarity changes** in all29,072 nonneutral
rate-utterance records. All seven current-availability patterns have exactly
unchanged W-F1. Mean8 W-F1 remains83.377462; high-missing W-F1 remains79.956560.
The original57 early A-only positive harm records have zero recoveries.

Under this checkpoint/protocol, the separately exposed root-specific component
has no detectable binary task contribution. This meets the predeclared
"unchanged scores:pause enlarging RER" condition. It does not establish that all
rooted computation is unnecessary:the retained public mean is itself computed
from the original rooted representations. Root-specific information may also
be redundant with the retained original node or public summary. A different
architecture trained from scratch is not evaluated here.

## Intervention and exact scope

Original: `o_i = readout([x_i, r_i, r_bar])`.

Intervention: `o_i = readout([x_i, r_bar, r_bar])`.

The temporary pre-hook changes only the middle third of the final core
readout input, copying the already-calculated last third. It explicitly checks
that the last third is the original mean of the rooted vectors. Original
projected node x_i, both untouched chunks, graph adjacency/subgraphs, root labels,
GIN layers, rooted pooling, weights, decoders, Local Skip, Flat and task head
are unchanged. The hook is removed even after exceptions.

Each batch runs one original OSRAM scan and captures the same Local/Base/Gap.
The input-side adapter/GNN is then replayed with the temporary hook; original
and common-root task inputs go through the same Flat/head. This performs two
GNN computations but only **one Memory trajectory**, not a second CHM forward.
There are176 original Memory forwards and985 intervened packed-core readout calls.

Model:MOSEI seed66, `mosei_f128_n10`:Flat hidden128/output1600, Nested dim64,
depth3, graph groups2, GIN MLP128; CHM remains8 actual heads x64, write_step0.6.
This is not the graph-groups8 N15 or the historical LargeFlat+Nested32 run.
The existing eight per-rate Test-best weights and exact GCNet masks are held
fixed, with no intervention-dependent checkpoint or threshold selection.

## Overall metrics

| Readout | Mean8 ACC | Mean8 W-F1 | High ACC | High W-F1 |
|---|---:|---:|---:|---:|
| Original |83.592460|83.377462|80.251330|79.956560|
| Common root |83.592460|83.377462|80.251330|79.956560|
| Difference, pp |0|0|0|0|

## Seven current-availability groups

These pool matching rate-utterance records across the eight rates. The same
utterance can appear repeatedly under different rates/checkpoints; counts are
not independent observations. Pooled group W-F1 is NOT mean8 W-F1.

"Original corrections" means original correct/common-root wrong; "original
harms" means original wrong/common-root correct. Both are zero in every group.

| Current pattern | N, nonneutral | Original W-F1 | Common-root W-F1 | Delta, pp | Original corrections | Original harms |
|---|---:|---:|---:|---:|---:|---:|
| A |3692|74.840040|74.840040|0|0|0|
| T |3717|87.192575|87.192575|0|0|0|
| V |3645|76.007013|76.007013|0|0|0|
| AT |2647|87.317401|87.317401|0|0|0|
| AV |2676|76.846996|76.846996|0|0|0|
| TV |2724|87.361088|87.361088|0|0|0|
| ATV |9971|87.343404|87.343404|0|0|0|

## Per-rate metrics

Each cell below is identical between original and common-root readouts.

| Missing rate | Selected N10 epoch | ACC | W-F1 | MSE |
|---|---:|---:|---:|---:|
|0.0|24|87.561915|87.369885|0.456420|
|0.1|24|86.901486|86.699260|0.497569|
|0.2|12|85.470556|85.402157|0.547415|
|0.3|24|84.837644|84.561335|0.568567|
|0.4|12|83.214089|83.117375|0.644670|
|0.5|12|81.975784|81.778345|0.674609|
|0.6|24|80.022014|79.739347|0.743712|
|0.7|24|78.756192|78.351989|0.788938|

Original MSE includes gold0; binary ACC/W-F1 exclude gold0 and use prediction>0.
All72 rate/group rows, including early_A_positive, are in results/PER_RATE.csv.

## Did the intervention actually run?

Yes. The maximum actual removed `|r_i-r_bar|` is0.0133624105 across real packed
test graph calls, not a synthetic input test. Some root-specific inputs differ;
this is not proof that all rooted vectors were equal.

Of37,272 total predictions including neutral labels,14,001 differ in floating
value after intervention; maximum absolute prediction difference is
4.76837158203125e-6 and mean absolute difference is1.692281982774104e-7.
No prediction changes binary polarity. Thus "same W-F1" does not mean the
intervention was skipped or that all prediction arrays are byte-identical.

The earlier57 Flat-correct/N10-wrong early A-only positive records (49unique
utterances) all remain wrong. The separate whole-Nested-off replay restored13;
that direct correction effect cannot be equated with the rooted-specific slot
examined here. Neither result identifies a unique cause of training drift.

## Verification and resources

- Four focused tests ran first in a missing-implementation red state, then
  passed in cmerc-repro:middle-chunk-only replacement, equal-root identity and
  invalid shape, actual Nested forward/state/RNG preservation, exception cleanup.
- Original replay predictions match the protected N10 predictions to1e-6 and
  have exact original polarity. Checkpoint/prediction hashes, config, environment,
  IDs/indices/labels/availability and fixed evaluation masks were checked.
- All frozen parameter hashes and original checkpoints are unchanged; first
  utterance predictions, causal backward zero halves and CPU/CUDA RNG are
  preserved. Temporary hooks are removed after each batch.
- Independent NumPy/sklearn verification recomputed per-rate ALL/seven-pattern
  metrics and all pooled groups from the saved NPZ files, confirming zero flips
  and the57-record/49-unique-utterance subcohort. See results/VERIFICATION.json.
- biggpu physicalGPU1, UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`;
  process3306548, tmux `yb_root_specificity_20261025`, cmerc-repro Python.
  Runtime90.820351seconds including setup. No optimizer or training update.
  GPU4 and other users' processes were not used or modified.
- The existing optional torch-sparse import warning occurred in the unchanged
  environment; this dense GIN path does not use that optional extension.
- No production model, missing sampler, dataset or prior result directory was
  edited. Only an independent experiment runner/hook/tests and small reports
  were added. Large predictions and weights are not committed.

## Reproduction and provenance

Immutable execution source commit:`e5f297048df29c828ab17aedb1046569d3cb55e3`.
Its archive contains1513 committed Python/AGENTS files; canonical source manifest
SHA256:`1e6522a8c917c977ceb054160267244d4328c79d531214d6fcc32461662fb21d`.

```bash
cd /data2/yb/remote_experiments/osram_mosei_root_specificity_20261025/source_v1
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_mosei_root_specificity_20261025.run \
  --source-commit e5f297048df29c828ab17aedb1046569d3cb55e3 \
  --output /data2/yb/remote_experiments/osram_mosei_root_specificity_20261025/replay_v1
```

The shown output already exists; runner refuses overwrites. An intentional
rerun must use a fresh output directory. GPU UUID is fixed before torch import.

Remote source/output/log all reside under:
`/data2/yb/remote_experiments/osram_mosei_root_specificity_20261025/`.

SUMMARY SHA256:
`f52d80964bbdada27698463bb56a37fa96e711ff5b743318247402e0ae8de06a`.
The separate results commit follows the fixed implementation commit above.

## Limits and next decision

Single seed and test-selected weights:internal model-behavior evidence only.
The retained public mean still includes root-conditioned graph computation;
this test does not remove all rooted processing, nor does it train a common-root
replacement model. It cannot prove causal semantics, statistical significance
or generalization benefit. It also cannot show that repairing root encoding
would improve task performance.

Pause enlarging the current RER on the basis of root-specificity. No further
training, seeds, gates, topology changes or hyperparameter searches are launched.
