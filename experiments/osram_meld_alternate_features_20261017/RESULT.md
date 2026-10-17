# MELD alternate features — MAGTKD completed, MM-Align blocked

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC.

| Feature source | Data state | Training state | ACC/UA |
|---|---|---|---|
| MAGTKD |Complete; canonical13,708utterances and17field conversion verified|Corrected v2 complete;50/50 epochs;verified 2026-10-17 07:31 UTC|Final eight-rate Test-best ACC54.631%,UA31.942%|
| MM-Align |dev/test/embedding verified; train incomplete due Drive quota; dev110_7absent|Not started; strict full-coverage guard|Unavailable|

No independent-test paper claim. No local training or environment change. The intended
only-data comparison is seed66/50epochs,original GCNet random missing and Flat.
User clarification is required before dropping any missing MM-Align utterance or
introducing a reduced-sample control; current code refuses both silently.

Prepared MAGTKD hash and provenance are in `MAGTKD_PREPARED.json`. Upstream
acquisition integrity/source notes are separate artifacts. The helper submitted
the job after SSH recovered. Its three-batch smoke completed,but the first
formal epoch hit a Torch1.13UUID RNG enumeration issue. The failed directory is
preserved. A process-local compatibility fix passed real GPU save/restore and
was deployed as a fresh v2run,not an incomplete-state continuation.

## Final MAGTKD seed66 scores

INTERNAL TEST-ORACLE ONLY. Each rate has its own ACC-selected checkpoint; this
is not the eight-rate deployment performance of one selected checkpoint. UA and
W-F1 use exactly the ACC-selected epoch's saved predictions and mask.

| Missing rate | Selected epoch | ACC (%) | UA (%) | W-F1 (%) |
|---|---:|---:|---:|---:|
|0.0|1|56.590|37.516|54.651|
|0.1|1|56.782|37.014|54.799|
|0.2|1|54.559|34.262|52.348|
|0.3|6|53.525|30.339|49.958|
|0.4|6|53.563|30.217|49.926|
|0.5|38|53.716|28.645|49.759|
|0.6|6|54.291|30.467|50.682|
|0.7|15|54.023|27.074|48.882|
|Eight-rate mean|—|54.631|31.942|51.376|
|High-missing mean (.5/.6/.7)|—|54.010|28.729|49.774|

Historical ECERC Flat seed66 reference: ACC60.675%,UA35.436%. MAGTKD differences:
-6.044pp ACC and -3.494pp UA. Not an exactly paired-source control (see RNG
and input-dimension caveats below). No downstream benefit under this fixed
configuration; upstream feature corruption is not established by these scores.
No additional training or tuning was launched.

Remote SUMMARY status complete and history contains50epochs; intended PID has
exited. Independently verified all8BEST checkpoint hashes,selected prediction
hashes,mask hashes,selected epochs and same-artifact ACC/UA/W-F1,with2610test
utterances per rate. Full recovery file is retained (161718909bytes).
Wall time2697.284seconds (~45minutes); peak CUDA allocated606427648bytes
(~578MiB,not total device usage);13455399parameters; no Block2parameters.
Machine-readable results/config/source/environment provenance:
`MAGTKD_FINAL_SUMMARY.json`. Raw weights and predictions remain remote:
`/home/qinjinghui/remote_experiments/osram_meld_alternate_features_20261017/runs/Flat_MAGTKD_seed66_v2`.

## Historical interim scores (21/50 epochs, not final)

Scores below use each rate's ACC-selected checkpoint and its exact saved
prediction artifact. UA is seven-class macro recall from the same predictions;
no new randomized inference was used. All eight artifact ACCs matched their
checkpoint selection metadata within 1e-8 (fraction units).

| Missing rate | Selected epoch | ACC (%) | UA (%) |
|---|---:|---:|---:|
| 0.0 | 1 | 56.590 | 37.516 |
| 0.1 | 1 | 56.782 | 37.014 |
| 0.2 | 1 | 54.559 | 34.262 |
| 0.3 | 6 | 53.525 | 30.339 |
| 0.4 | 6 | 53.563 | 30.217 |
| 0.5 | 13 | 53.333 | 28.048 |
| 0.6 | 6 | 54.291 | 30.467 |
| 0.7 | 15 | 54.023 | 27.074 |
| Eight-rate mean | — | 54.583 | 31.867 |

Historical ECERC Flat seed66 reference: ACC60.675%,UA35.436%.
Current differences are -6.092pp ACC and -3.569pp UA. This is a descriptive
historical comparison,not an exact paired-source control: the corrected CUDA
RNG shim and input dimensions prevent asserting identical initialization/masks.
The run was kept at the authorized 50-epoch budget. No tuning,early stop or
additional control was launched. Low interim scores do not identify their cause.

At 37/50 epochs the selected artifacts/epochs above remained unchanged: no
ACC-selected best improved since the previous 21-epoch check. Eight-rate W-F1
is51.312%; high-missing ACC53.883%,UA28.530%,W-F1 49.604%. The recovery checkpoint
existed (161684925bytes). These were interim,not final results.

Follow-up feature integrity/semantics check: see FEATURE_CHECK.md. All41,124
converted selected-stream vectors match source arrays exactly; no MAGTKD
nonfinite/zero vectors or near-constant channels found. This does not establish
upstream extraction validity or explain the low downstream scores. No run change.
