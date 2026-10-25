# Early A-only positive harms: same-weight Nested bypass

INTERNAL DIAGNOSTIC ONLY. MOSEI seed66, Flat128 versus N10, cmerc-repro,
gcnet-random-mask-v1, original eight per-rate Test-best checkpoints/masks.
No training, model forward, mask generation or GPU use. Only 88 already-saved
prediction files were read; their SHA256 and identity arrays were checked.
All protected models, masks, predictions, datasets and running jobs are unchanged.

## Fixed cohort and result

Reproduces the previously reported current A-only, gold-positive, turns2–3
cohort:581 rate-utterance records,5 corrections and57 harms versus independently
trained Flat128. The57 harms involve49 unique conversation/utterance identities.
Saved utterance indices are zero-based; indices1/2 mean turns2/3.

Interventions use exactly the same Nested weights and test mask, bypassing only
the extra task-input corrections. Original Local Skip, CHM queries/writes and
original Base/Gap reads are retained. This does NOT delete original Memory.

| Same-weight readout | Restored among the57 harms | Still nonpositive |
|---|---:|---:|
| Full Nested |0|57|
| All Nested corrections off |13|44|
| All corrections halved |5|52|
| Only Local correction off |1|56|
| Only Base correction off |2|55|
| Only Gap-A correction off |0|57|
| Only Gap-T correction off |4|53|
| Only Gap-V correction off |1|56|

Individual recoveries overlap and must not be added. The union of all tested
off cases restores13; all individual-slot recoveries are among those13.
Gap-A is inactive in A-only:polarity is unchanged; raw float discrepancies of
approximately1e-9 in its aggregate prediction shift are roundoff, not evidence
of an active Gap-A path.

All57 predictions increase when all corrections are bypassed (mean increase
0.052738);13cross zero. Thus the direct branch negatively shifts all57 selected
positive targets under this intervention. The44 remaining errors do NOT prove
the branch has no effect:they also move upward but remain nonpositive.
They DO show that the current direct branch cannot alone explain the complete
cross-model polarity difference. Jointly trained main-path parameters and
different selected epochs remain confounded; write/read/Flat are not isolated.

## Do not optimize only the selected mistakes

| Cohort, all-corrections-off versus full N10 | Corrections | Harms | Net correct |
|---|---:|---:|---:|
| Early A-only positives, N581 |13|0|+13|
| Early A-only nonneutral, N782 |13|5|+8|
| All eight-rate nonneutral records, N29072 |91|82|+9|

Five early negative records become incorrect. Across the whole test replay,
the previously saved eight-rate W-F1 is83.366358 with all corrections off,
versus83.377462 with full Nested (−0.011104pp), despite net+9correct. Net counts
are not a W-F1 decomposition and cannot be substituted for the accepted metric.
This does not support deploying an early-turn or test-label-selected bypass.

## Per-rate localization of the original57 harms

| Rate | Original harms | Restored by all off | Remain | Flat / Nested selected epoch |
|---|---:|---:|---:|---|
|0.0|0|0|0|21 /24|
|0.1|0|0|0|23 /24|
|0.2|5|0|5|16 /12|
|0.3|6|3|3|16 /24|
|0.4|7|1|6|15 /12|
|0.5|8|1|7|16 /12|
|0.6|9|3|6|19 /24|
|0.7|22|5|17|16 /24|

## Decision and limits

- Do not attribute these57 A-only harms to the separately observed ATV clique
  rooted-pool degeneracy. They occur under a different graph condition.
- Do not increase graph depth, repeat correction-strength search or add an
  early-history Gate based on this post-selected error cohort.
- A root-encoding repair remains an independently testable architecture idea,
  not an established remedy for these errors. This analysis does not train it.
- The evidence now separates a direct negative contribution on selected
  mistakes from a larger residual cross-model difference. It does not identify
  a unique training mechanism responsible for the latter or prove that an
  alternative training procedure would help.

## Reproduction and checks

Run on biggpu, CPU only, using the saved remote predictions:

```bash
CUDA_VISIBLE_DEVICES='' /data2/yb/conda_envs/cmerc-repro/bin/python \
  experiments/osram_mosei_n10_early_harms_20261025/analyze.py \
  --source-commit 02b21f6701001958dcc3b7e37a3fb7b5a981b19d \
  --output /data2/yb/remote_experiments/osram_mosei_n10_early_harms_20261025/output_v1
```

The output directory must not exist; the script refuses overwrites. The shown
output already exists from the completed analysis; use a fresh directory for
an intentional rerun. Exact executed script SHA is recorded in SUMMARY.json.

- Three synthetic unit tests passed:label0 exclusion/zero prediction boundary,
  alignment rejection, early-index scope and overlapping recoveries.
- All88 input prediction SHA256 match their original manifests; both replay
  manifests point to the same Nested/Flat summaries; all rate checkpoint hashes
  match. Original Flat/Nested speaker arrays match; replay files do not contain
  speakers, so they use exact IDs/indices/labels/availability alignment.
- Both replay gamma1 predictions match original Nested to1e-6 and exact polarity.
- A separate per-record loop independently reproduced57records,49unique IDs,
  13recoveries and every per-rate count. Whole-replay91/82 reproduces the prior
  completed global-strength audit.
- Initial input validation rejected original predictions stored as[N,1]; the
  reader now explicitly canonicalizes only[N,1] to[N]. No predictions changed.
- No full-model tests or fresh inference were needed or claimed.

Remote output:
`/data2/yb/remote_experiments/osram_mosei_n10_early_harms_20261025/output_v1`

SUMMARY.json SHA256:
`85108c17b397a26cc1b79f1ba7a37dde00cd31c13d81ff7599c5355f9494d639`

The parent source revision is02b21f6; this task's final commit contains the new
analysis script, tests and reports. No numerical model code was modified.
