# Current status — not training results

INTERNAL TEST-ORACLE FEATURE-SOURCE DIAGNOSTIC.

| Feature source | Data state | Training state | ACC/UA |
|---|---|---|---|
| MAGTKD |Complete; canonical13,708utterances and17field conversion verified|Corrected v2 running on GPU3;37/50 epochs at 2026-10-17 07:12 UTC;no error|Interim eight-rate Test-best ACC54.583%,UA31.867%|
| MM-Align |dev/test/embedding verified; train incomplete due Drive quota; dev110_7absent|Not started; strict full-coverage guard|Unavailable|

No performance claim. No local training or environment change. The intended
only-data comparison is seed66/50epochs,original GCNet random missing and Flat.
User clarification is required before dropping any missing MM-Align utterance or
introducing a reduced-sample control; current code refuses both silently.

Prepared MAGTKD hash and provenance are in `MAGTKD_PREPARED.json`. Upstream
acquisition integrity/source notes are separate artifacts. The helper submitted
the job after SSH recovered. Its three-batch smoke completed,but the first
formal epoch hit a Torch1.13UUID RNG enumeration issue. The failed directory is
preserved. A process-local compatibility fix passed real GPU save/restore and
was deployed as a fresh v2run,not an incomplete-state continuation.

## Interim scores (not final)

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
The run remains at the authorized 50-epoch budget. No tuning,early stop or
additional control was launched. Low interim scores do not identify their cause.

At 37/50 epochs the selected artifacts/epochs above remain unchanged: no
ACC-selected best improved since the previous 21-epoch check. Eight-rate W-F1
is51.312%; high-missing ACC53.883%,UA28.530%,W-F1 49.604%. The recovery checkpoint
exists (161684925bytes). These are still interim,not final results.
