# 150-epoch completion: final results

**INTERNAL DIAGNOSTIC ONLY** — per-rate Test-oracle BEST, not independent held-out model selection.

## Completion and protocol

47/47 authorized runs completed; zero failed/pending/running. The 15 IEMOCAP4 Flat runs remain cancelled. GPU7 drained its existing jobs without replacements; GPU1/2/3 used up to five concurrent jobs each. Completion timestamp: 2026-10-11 07:20:03 UTC.

Training is 100 epochs at constant LR 0.001, then 50 epochs at constant LR 0.0001, not 150 epochs at the original LR. Compatible full model/optimizer/RNG/training states were resumed, not BEST weights alone. Thirty Nested fold runs resumed existing epoch100 states; seventeen Flat runs required a fresh 100-epoch stage before continuation. Architecture, data and protocol provenance are in MANIFEST.json and per-run PROVENANCE.json. Launch code: 4eed138; scheduling revisions: dc84f75, bb04539. Scheduling interruption/recovery is documented in SCOPE47.md and RECOVERY_SCOPE47.json.

IEMOCAP: equal five-fold average, then eight rates, then seeds66/67/68; checkpoint selection ACC. UA is mean per-class recall recomputed from saved predictions, not macro-F1. MOSI: eight-rate mean, seeds66/67/68; selection W-F1. High missing means rates .5/.6/.7. All values below are percent.

## MOSI: matched 150-epoch results

| Seed | Flat ACC | Nested ACC | Flat W-F1 | Nested W-F1 | Nested−Flat W-F1 | Flat high W-F1 | Nested high W-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
|66|81.174|81.231|81.068|81.163|+0.095|76.352|76.077|
|67|81.155|81.021|81.119|80.851|−0.269|77.034|75.497|
|68|80.107|79.707|80.053|79.674|−0.379|74.440|74.314|
|Mean|80.812|80.653|80.747|80.562|−0.184|75.942|75.296|

This batch adds Flat67/68. Flat66 and Nested66/67/68 reuse the completed lower-LR150 results in ../osram_nested_low_lr_extend150_20261010/{flat,nested,nested67,nested68}_FINAL_METRICS.json. Do not confuse them with the separate constant-LR150 experiment. FINAL_SUMMARY.json's CMUMOSI_flat group contains only the two newly completed seeds, not this three-seed mean.

Flat67 W-F1 increased from 80.556 to 81.119 (+0.563 pp); Flat68 stayed at 80.053. Their fresh epoch100 aggregate scores reproduce the historical100 scores. Nested does not outperform Flat across these three seeds.

## IEMOCAP: per-seed 150-epoch results

| Dataset/model | Seed66 ACC / UA | Seed67 ACC / UA | Seed68 ACC / UA | Mean ACC / UA | High ACC / UA |
|---|---:|---:|---:|---:|---:|
|4-class Nested|79.509 / 79.720|78.747 / 79.195|79.317 / 79.819|79.191 / 79.578|77.108 / 77.466|
|6-class Flat|61.415 / 60.178|61.433 / 59.328|61.089 / 59.389|61.312 / 59.632|59.113 / 57.266|
|6-class Nested|60.889 / 59.520|61.812 / 60.463|61.738 / 60.484|61.480 / 60.156|59.334 / 57.915|

6-class Nested−Flat: ACC +0.168 pp, UA +0.524 pp. Seed66 is worse, while seeds67/68 improve: not a uniform seed-level gain.

4-class Flat150 was cancelled by the user. Historical Flat100 was ACC79.214 / UA79.478; this is not a matched-budget Flat150 comparator.

## Continuation gains against the actual source100

| Run family | Source100 ACC / UA | Final150 ACC / UA | Delta ACC / UA (pp) |
|---|---:|---:|---:|
|4-class Nested|79.054 / 79.477|79.191 / 79.578|+0.137 / +0.101|
|6-class Flat|61.288 / 59.599|61.312 / 59.632|+0.024 / +0.033|
|6-class Nested|61.469 / 60.119|61.480 / 60.156|+0.011 / +0.037|

The fresh 6-class Flat100 differs from historical Flat100 (61.221 ACC / 59.901 UA); historical runs lacked the explicit selection-metric setting. Do not attribute that rerun difference to the extra50 epochs or Nested. Continuation yields small IEMOCAP changes. Cumulative BEST retains epoch1–100 candidates; expanding the selection window cannot lower its selection metric and is not itself evidence of generalization improvement.

## Verification and artifacts

FINAL_SUMMARY.json contains all47 per-run/per-rate metrics, source100 and historical comparator scores, prediction SHA256, paths, final queue and grouped statistics. No new inference or training was used for this summary. summarize.py verified complete provenance, 150 history entries, 50 LR traces at1e-4, existence of required artifacts and hashes of non-weight outputs; it did not rehash every checkpoint. IEMOCAP prediction ACC agrees with saved metrics and every class has evaluation support.

Remote root: /home/qinjinghui/yb_gcnet_experiments/osram_complete150_20261011

Reproduce summary on biggpu:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python summarize.py --root /home/qinjinghui/yb_gcnet_experiments/osram_complete150_20261011
```

No checkpoints are uploaded to Git. No further training was launched by the status/report task.

## Seed67 BEST timing check

Rechecked saved selected_epoch_by_rate (rates0.0 through0.7):

- Nested150: [55,47,47,47,47,47,48,49]. Its 80.851 eight-rate W-F1 was already attained by epoch55; the extension did not improve it.
- Flat source100: [69,70,70,70,72,70,87,72].
- Flat150: [69,107,70,107,109,70,107,103]. Its 81.119 final aggregate was already attained by epoch109, not epoch150; epochs110–150 provided no further BEST improvement.

Sources: local nested67_FINAL_METRICS.json referenced above and remote runs/CMUMOSI_flat_s67_f1/{stage100,stage150}/metrics.json. “150-epoch result” denotes the completed training budget, not the epoch when the selected checkpoints were obtained. These are eight separately selected per-rate checkpoints from one training trajectory, not a single best epoch.
