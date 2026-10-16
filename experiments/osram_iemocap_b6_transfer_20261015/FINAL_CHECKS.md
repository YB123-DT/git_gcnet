# Final verification —2026-10-16

INTERNAL TEST-ORACLE SCREEN. All30 authorized fold runs completed100epochs.
The queue and persistent controller both report complete; no failed or pending
jobs remain. No new training or inference was launched for this verification.

Verified directly on biggpu:

-30 complete worker summaries and30 histories, totaling3000 completed epochs.
-All configs retain gcnet-random-mask-v1.
-240 nonempty BEST checkpoint files remain present (8 per fold run).
-240 exact selected-epoch prediction artifacts match their recorded file SHA256.
-Recomputed ACC/UA/W-F1/CE from all240 stored artifacts match reported metrics
  within1e-6; selected epochs and argmax predictions match their saved arrays.
-300 CSV metric rows and both complete fivefold result groups are present.
-Each dataset has seeds66/67/68 with folds1–5; reported three-seed means and
  sample SDs were independently recomputed for overall and high-missing metrics.

## Complete fivefold results

Equal-fold, then equal-rate means (%); three-seed uncertainty is sample SD.

| Dataset | Seed | ACC | UA | W-F1 |
|---|---:|---:|---:|---:|
| IEMOCAP4 |66|79.408|79.655|79.263|
| IEMOCAP4 |67|79.155|79.474|79.034|
| IEMOCAP4 |68|79.298|79.564|79.184|
| IEMOCAP4 |Mean±SD|79.287±0.127|79.564±0.090|79.160±0.117|
| IEMOCAP6 |66|61.637|59.819|61.239|
| IEMOCAP6 |67|61.746|60.392|61.462|
| IEMOCAP6 |68|61.818|60.147|61.350|
| IEMOCAP6 |Mean±SD|61.734±0.091|60.119±0.287|61.350±0.111|

These full fivefold results must not be confused with the earlier fold5-only
ACC82.830/64.944. Complete rate/fold/seed metrics remain in SUMMARY.csv/JSON;
FOLD5.json keeps that separate subset unchanged. Historical exact-selection
Flat UA is unavailable, so no fabricated paired Flat-UA comparison is made.

No changes to models, missing masks, optimizers or selected B6 parameters were
made during this final check. Success means the bounded transfer experiment
and its evidence are complete, not that the method has improved a baseline.
