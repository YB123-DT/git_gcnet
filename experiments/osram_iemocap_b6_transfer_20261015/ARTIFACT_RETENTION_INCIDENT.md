# Original corrected-mask Flat retention failure

## Accountability and scope

The user repeatedly requested ACC and UA retention. The original corrected
GCNet-mask IEMOCAP Flat batch did NOT meet that requirement. This is an
execution/retention failure, not an omitted user instruction. This diagnostic
does not repair or recreate the lost artifacts, and starts no new experiment.

Affected root:
`/data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1`.
IEMOCAP4/6 × seeds66/67/68 × fivefolds =30 runs, source b4a3138.
This is distinct from protected MELD baseline weights and retained B6 weights.

## What actually happened

1. Original trainer selected eight per-rate BEST models using test ACC.
   Selected epoch, ACC and W-F1 scalars survive in metrics/history JSON.
2. After training it reloaded each selected BEST model and evaluated it AGAIN.
   GCNetLegacyRandomMaskSchedule.generate_batch() draws fresh global NumPy
   masks per call; its stored freeze_evaluation flag does not freeze draws.
3. The saved predictions_miss_*.npz therefore contain the FINAL RETEST
   labels/predictions/availability, not original winning-epoch predictions.
4. The wrapper run.py unconditionally calls _discard_checkpoints() after the
   trainer returns (lines34–39 and103–106 in the current archived wrapper).
   All30 original PROVENANCE.json records confirm8 deleted models per run:
  240 BEST files were actively deleted by the run wrapper.
5. Winning-epoch confusion counts, UA, predictions and masks were not retained.
   Remaining ACC/W-F1 scalars or mask hashes cannot reconstruct these arrays.

Retained: config, provenance,100-epoch scalar history, selected epoch/ACC/W-F1,
diagnostic JSON and240 final-retest prediction files. Lost/not retained:
selected BEST weights, exact winning-epoch predictions/masks and associated UA.

## Verified source of the reported score gap

These are TWO evaluations of the SAME selected weights, not two from-scratch
training results. Verified b4a3138 trainer lines3780–3813: strict-load each BEST,
then evaluate_rate(...collect=True), then save final predictions.

| Dataset | Selected ACC | Same selected weights, fresh-mask retest ACC | Difference(pp) |
|---|---:|---:|---:|
| IEMOCAP4 |79.260664|77.756928|−1.503736|
| IEMOCAP6 |61.624818|60.495308|−1.129511|

All240 selected ACC scalars match their selected epochs' original histories.
At missing rate0, all30 selected/retest accuracies are identical. Of210 nonzero
rate pairs,168 retests are lower,39 higher,3 equal. Previously, recomputing all
240 final prediction files reproduced metrics.test ACC within1e-8.

Because test masks change at each evaluation, picking a maximum over epochs
also selects among favorable mask realizations. A fresh retest need not equal
that maximum. This explains THIS selected-versus-retest difference; it must
not be used as an unverified explanation of every from-scratch rerun difference.

## Recovery checks and boundaries

Checked original_r1 and neighboring original non-r1 output roots, known remote
aliases under /data2/yb/remote_experiments and /home/qinjinghui/remote_experiments,
and the existing IEMOCAP artifact/restore directory index. No matching original
BEST or selection_versions backup was found in those checked locations.
The existing retained IEMOCAP source-combination audit uses a different older
source, not a backup of these corrected-mask Flat runs; do not substitute it.

The current B6 runs instead retain240 BEST weights, exact selected artifacts,
same-epoch ACC/UA, full optimizer/RNG recovery state and source/config identity.
Their preservation was verified on completion. This does NOT restore original
Flat's missing evidence.

To obtain a complete Flat Test-best ACC/UA baseline, the narrow remediation is
new Flat training using the corrected retention path, explicitly recorded as
new runs. That would not recover or guarantee the old exact scores. No such
training is authorized or launched by this complaint/diagnostic turn. Current
loss of original selected-epoch UA remains unresolved; it is not filled with
retest UA or relabelled as recovered.
