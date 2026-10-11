# Persistent A-only: where the positive bias appears

INTERNAL DIAGNOSTIC ONLY. MOSI seed66; three frozen miss=.7 BEST checkpoints.

## Observed failure location

The rho.10 trained Flat readout predicts positive for every nonneutral A-only utterance when Base/Gap inputs are zero. Adding persistent Audio-only history changes only one polarity. On matched current-A utterances, original random-history reads substantially compensate this positive bias. The failure therefore already exists in the no-history-evidence decision path; it is not created solely by adding Gap reads.

This localizes behavior to the learned Local-to-task decision path under absent historical reads. It does not separately identify an encoder defect versus a classifier defect, or prove Audio has no sentiment information.

## Persistent A-only: all656 nonneutral utterances

L is the SAME trained Flat with Base=Gap=0, not a separately trained Local model. LB masks Gap; LG masks Base; LBG is unchanged full output.

| Model | Readout | Predicted positive /656 | W-F1 % | Prediction mean* | Prediction std* |
|---|---|---:|---:|---:|---:|
|Flat|L|631|27.986|0.5425|0.2856|
|Flat|LB|592|35.378|0.4086|0.2923|
|Flat|LG|631|27.986|0.5426|0.2860|
|Flat|LBG|591|35.643|0.4087|0.2926|
|rho050|L|208|57.788|-0.3506|0.6081|
|rho050|LB|155|56.863|-0.4860|0.6023|
|rho050|LG|208|57.788|-0.3517|0.6074|
|rho050|LBG|154|56.986|-0.4870|0.6017|
|rho010|L|656|25.073|0.8448|0.2441|
|rho010|LB|655|25.404|0.7927|0.2551|
|rho010|LG|656|25.073|0.8445|0.2409|
|rho010|LBG|655|25.404|0.7925|0.2518|

*Prediction moments include all686 utterances; classification excludes30 neutral labels.

## Same current A utterances: paired history comparison

161 nonneutral utterances, identical IDs/labels/current raw inputs. Local differences are only float32 packing noise (absolute≤1.91e-6, relative≤1.93e-7). No model parameters changed.

| Model | History | Readout | Predicted positive /161 | W-F1 % |
|---|---|---|---:|---:|
|Flat|random|L|155|26.263|
|Flat|random|LBG|65|65.217|
|Flat|persistent|L|155|26.263|
|Flat|persistent|LBG|145|34.783|
|rho050|random|L|53|58.127|
|rho050|random|LBG|21|59.562|
|rho050|persistent|L|53|58.127|
|rho050|persistent|LBG|43|59.548|
|rho010|random|L|161|23.223|
|rho010|random|LBG|71|65.424|
|rho010|persistent|L|161|23.223|
|rho010|persistent|LBG|160|24.556|

## What is established and what is not

- rho.10: all656 positive without read values;655 positive with full A-only history. Base/Gap removal does not rescue this model.
- rho.50:208 positive without read values,154 positive with full A-only history. Its better fixed-pattern output does not require a positive contribution from persistent Memory in this readout intervention.
- Original Flat is also biased positive without reads (631/656); rho.10 intensifies this observed weakness. This was not introduced by InfoNCE: both compared paired-view models are task-only.
- On matched current-A samples, rho.10 Full scores65.424 with random history versus24.556 with A-only history, while Local-input readout is23.223 in both. Random test performance can conceal failure when historical support is persistently restricted.
- First turns provide a supplementary natural no-history check: rho.10 predicts all26 nonneutral first turns positive, and L/Full outputs agree. Small cohort, not a separate broad claim.
- Random-history versus persistent-A changes historical support in both directions: it removes T/V but can restore earlier A. It is not a pure Text-deletion intervention and cannot identify Text alone as the causal source.
- Setting active reads to zero may be out-of-distribution. Zero-readout outcomes cannot establish absence of information in Local, and the first-turn check has limited sample size.
- Checkpoints differ (Flat39, rho.50 epoch43, rho.10 epoch100). No common-epoch A-only trajectories are available, so weaker augmentation weight versus late training/selection cannot be causally separated here.
- Historical code/config audit finds no reversed loss weights, mask difference, modality swap, inference second view or shared Memory. See CODE_AUDIT.md.
- No training, threshold search, learned routing, new block or parameter change. One backbone seed, inherited Test-oracle protocol.

## Execution and verification

Historical backbone6e2a998; runner61d011f. biggpu GPU6, Python PID1805989.
Remote root: /data1/yb/remote_experiments/osram_paired_aonly_localization_20261011.
Initial run stopped on overly strict absolute Local tolerance1e-6. Reused the already validated historical diagnostic limits (absolute1e-5 AND relative1e-6), with exact current raw input equality. Original failure/log preserved. No model or input change.
All three original random prediction archives and full replay outputs match exactly; checkpoint/model hashes unchanged. All per-utterance values and all grouped readouts retained in results_v2/.

```bash
CUDA_VISIBLE_DEVICES=GPU-e4cafb17-818e-216a-b94a-7440063a9153 python -m experiments.osram_paired_aonly_localization_20261011.run \
  --output <new-output-directory> --commit 61d011f
```
