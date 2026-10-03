# Gap increment audit — original cfg84 seed66

INTERNAL DIAGNOSTIC ONLY

## Stage1 completed

Eight original Full per-rate BEST checkpoints, not separately trained ablations.
Exactly eight causal scans (one test batch per rate) supplied the unchanged Local,
Base, masked Gap and Local Skip for all three classification readouts. No training.
All saved Full predictions match the previous original predictions exactly (max
absolute error0); masks, labels, polarities and checkpoint tensors/hashes were
checked. First valid utterance and fully observed readout equalities passed.

|Readout|8-rate W-F1|High-missing W-F1|
|---|---:|---:|
|Local only, P_L|74.820|67.503|
|Local + Base, P_B|79.439|73.865|
|Original Full, P_F|81.068|76.352|
|Gap increment, P_F minus P_B|+1.629|+2.487|

Gap rescue/harm:250/155 across5,248 nonneutral rate-exposures; high missing143/88.
There are4,010 both-correct and833 both-wrong exposures. All5,488 valid utterance
records, including neutral labels and per-utterance squared-error differences,
are preserved. Repeated rates are not independent samples.

## Observable evidence

Primary signed AUROC is computed per rate then macro-averaged over rates with
both rescue and harm. Rescue is positive; score direction is never optimized.

|Observable|Rescue mean|Harm mean|Raw AUROC|
|---|---:|---:|---:|
|Local margin|0.603|0.566|0.530|
|Base norm|16.520|15.650|0.552|
|Gap/Base norm ratio|1.429|1.349|0.537|
|Mean active Base/Gap cosine|0.829|0.795|0.557|
|Mean active residual query cosine|0.815|0.802|0.517|
|Mean active rho|0.815|0.802|0.517|
|Mean active eta|0.285|0.302|0.480|

These are weak aggregate separators, not proof of no information. For example,
Base/Gap cosine AUC across rates.1–.7 is.536,.680,.514,.672,.523,.451,.519:
the association is not consistently strong. Current-availability strata show
some exploratory signals: AV cosine macro AUC.644 (60 rescue/38 harm exposures),
A-only ratio.625 (90/52), but A-only cosine.447 and V-only ratio.470.
T-only AUC1.0 is based on extremely sparse harm cases (only1 harm total) and must
not be promoted as a reliable discriminator. AT/TV have no rate with both classes.
See per-rate and availability tables for precise valid counts and undefined cases.

No Gate or threshold is fitted. This evidence does not justify a general
norm/cosine Gate, but it also does not establish that all internal observables
are useless. Because the overall separation is weak/inconsistent, the requested
conditional stage2 head input ablation was run; no query changes.

## Stage2 completed

Contribution = W-F1(Full) minus W-F1(mask one forward64-d head slice).
Gap intervention masks the same head in all three already-masked Gap slots.
Head indices below are zero-based. Contributions need not sum to the all-Gap effect.

|Head|Base8-rate contribution|Gap8-rate contribution|Gap high-missing contribution|
|---|---:|---:|---:|
|0|+0.013|+0.081|+0.225|
|1|-0.036|-0.018|-0.097|
|2|-0.009|+0.002|+0.073|
|3|+0.051|+0.009|+0.128|
|4|-0.069|+0.066|+0.302|
|5|+0.006|+0.115|+0.491|
|6|+0.101|+0.057|+0.208|
|7|+0.348|+0.160|+0.640|

The requested separate Base/Gap by current-availability matrices are preserved
in `head_analysis/RESULT.md` and `head_analysis/matrix.csv` (including high missing).
Head7 Gap contributions at rates0.1–0.7 are -0.167,-0.193,-0.030,-0.252,
+0.544,+1.162,+0.215 overall. Its A/V/AV subgroup signs also change across rates.
This shows heterogeneous intervention responses but not a stable target-specific
specialization. No head Gate is justified by these numbers alone.

Stage2 used eight additional original trajectories and16 classifier-only head
readouts per batch. Its original three-readout CSV is byte-identical to Stage1,
and original Full replay remains exact. Six runner tests and13 analysis tests pass.
Inference code:565d445; remote code snapshot:code_heads; outputs:results_heads.

Because head effects do not yet establish stable specialization, the conditional
third-stage query/read cosine audit is being prepared without changing addressing.

## Reproduction and artifacts

- Inference implementation commit:26d11cb.
- Server:biggpu, host GPU5; remote root:/data2/yb/remote_experiments/osram_gap_increment_audit_20261003.
- Source:/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66.
- `results/PROVENANCE.json`: environment, code/input hashes, status, scan count.
- `results/checks.json`: each checkpoint hash, epoch and exact Full replay checks.
- `results/utterances.csv`: raw predictions and observables.
- `analysis/RESULT.md`: complete per-rate/observable table.
- `analysis/observables_per_rate.csv`: raw per-rate and availability-stratified AUC.

```bash
# Within the isolated remote snapshot, with original dataset environment:
CUDA_VISIBLE_DEVICES=5 python experiments/osram_gap_increment_audit_20261003/run.py \
  --source /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66 \
  --output NEW_OUTPUT --commit 26d11cb --gpu 5
python experiments/osram_gap_increment_audit_20261003/analyze.py \
  --input NEW_OUTPUT/utterances.csv --output NEW_ANALYSIS
```

Four runner tests (including actual-model instrumentation equality) and seven
offline-analysis tests passed before inference. OSRAM/model/training source files
were not changed. Results are Test-oracle internal evidence, not paper claims.
