# Interpretation and completion evidence

INTERNAL DIAGNOSTIC ONLY — MELD seed66; Test-oracle checkpoints, not a new trained method.

## Decisions supported by this diagnostic

1. B1's whole residual contributes a small net classification gain in its own checkpoint, not zero contribution. Across eight rate evaluations it corrects55 and harms41 cases (14 net). Its mean residual/anchor norm ratio is0.255279%. Opening it increases mean CE by0.001478 despite improving ACC and UA. High-missing ACC decreases0.025543pp. This is weak and mixed evidence, not a clear new historical-information gain.
2. At miss0.1 the large B1/B2 UA difference is mostly an epoch-selection effect in this paired replay: B2 epoch10→13 changes UA by+2.863017pp with ACC−0.038314pp. At epoch13 on the same mask, B2−B1 is UA−0.041408pp, ACC−0.038314pp. Do not generalize this one-rate comparison to all rate-selection effects.
3. B2's entire branch corrects44 and harms44 cases; ACC is unchanged, UA increases0.154350pp and CE worsens0.001158. The learned S has some continuous ranking signal, but no observed downstream advantage over S=.5: on−half ACC−0.004789pp, UA−0.004442pp and WF1−0.004471pp; zero corrections, one harm, three class changes across20880 rate-utterance evaluations.

Do not expand S/U or launch more seeds based on these results. Current evidence does not establish useful S-conditioned decision changes. It also does not prove that direct history retrieval or same-class relations can never help: the whole residual is very small, its MLP also uses Local/Base/Gap, and no isolated E/R intervention was requested.

## Whole residual on minus off

Off uses the SAME trained model's original Flat path, not separately trained original Flat.

| Model | ΔACC(pp) | ΔUA(pp) | ΔW-F1(pp) | ΔCE | Corrected | Harmed | Residual/anchor |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0.067050 | 0.254387 | 0.165056 | 0.001478 | 55 | 41 | 0.255279% |
| B2 | 0.000000 | 0.154350 | 0.089933 | 0.001158 | 44 | 44 | 0.240445% |

Counts sum8rate evaluations of2610 utterances each, not20880 unique utterances. Norm ratios are mean per-utterance ratios, not a ratio of mean norms.

## Per-rate direct effects

| rate | B1 ΔACC(pp) | B1 ΔUA(pp) | B1 ΔCE | B1 rescue/harm | B2 ΔACC(pp) | B2 ΔUA(pp) | B2 ΔCE | B2 rescue/harm | Learned S vs half rescue/harm |
|---|---:|---:|---:|---|---:|---:|---:|---|---|
| 0.0 | 0.306513 | 0.392130 | 0.001218 | 9/1 | 0.229885 | 0.345219 | 0.000921 | 8/2 | 0/0 |
| 0.1 | 0.153257 | 0.262404 | 0.002178 | 6/2 | 0.038314 | 0.083862 | 0.001175 | 3/2 | 0/0 |
| 0.2 | -0.038314 | 0.051805 | 0.000950 | 3/4 | -0.114943 | -0.004088 | 0.000960 | 1/4 | 0/0 |
| 0.3 | 0.000000 | 0.096651 | 0.001934 | 4/4 | -0.076628 | 0.025577 | 0.001668 | 3/5 | 0/1 |
| 0.4 | 0.191571 | 0.514277 | 0.001973 | 14/9 | 0.229885 | 0.384114 | 0.001845 | 10/4 | 0/0 |
| 0.5 | -0.153257 | 0.041845 | 0.000186 | 2/6 | -0.038314 | 0.066985 | 0.000118 | 3/4 | 0/0 |
| 0.6 | -0.153257 | 0.023432 | 0.001360 | 4/8 | -0.306513 | -0.057969 | 0.001266 | 3/11 | 0/0 |
| 0.7 | 0.229885 | 0.652551 | 0.002024 | 13/7 | 0.038314 | 0.391100 | 0.001307 | 13/12 | 0/0 |

## S prediction and calibration

Main relation statistics pool97136 pair evaluations; each rate has12142 strict-past pairs and the same35.900181% positive prevalence. Pairs are correlated within conversations and repeated across rates, so pooled metrics are descriptive, not independent-sample statistical evidence.

- Mean S0.403207, SD0.036566; 5/50/95 percentiles0.339505/0.410997/0.448944; range0.205589–0.573010. No probabilities≤.1 or≥.9.
- Only322/97136 probabilities are≥.5 (0.331494%). With the fixed threshold, positive recall is0.702569%, balanced ACC50.289451%, overall ACC64.272772% vs all-negative majority64.099819%. This is mostly predicting different-class at the fixed threshold.
- Pooled AUROC0.584116 and AUPRC0.460492 (positive prevalence0.359002); per-rate AUROC0.552330–0.665006. There is weak continuous ranking signal, not an entirely constant/no-information predictor.
- BCE0.650076, Brier0.228795, ECE10=0.045792; S=.5 constant gives BCE0.693147/Brier0.25. Beating a constant.5 baseline alone does not establish useful pair-specific learning, because positives are only35.9%. Calibration and downstream contribution must also be considered.
- R-weighted secondary AUROC0.583174; this does not change the downstream conclusion. No threshold fitting, calibration fitting or relation label input was used.

S=.5 holds E/R fixed, forces D=0 and rho=.5 for nonempty history; it is a frozen internal-distribution sensitivity intervention, not a newly trained benchmark.

## Completion, provenance and costs

- Diagnostic source commit: 470e653e957963f1cc291494ab486d365ac3de6f. Original numeric code b4a3138; B1/B2 extension442596a; same cmerc-repro runtime and accepted MELD features.
- B2 epoch13 belongs to the same complete50epoch B2 trajectory: config, branch specification, summary/final-state identity, saved BEST metadata and semantic state hash all match its committed epoch13 model version. It is sourced from miss0.5 BEST, not trained for miss0.5 separately.
- Sixteen original selected-checkpoint replays exactly reproduce saved logits (maximum error0), labels, IDs, speaker and availability. Manual same-scan readout on reproduces classes exactly; largest manual logit discrepancy1.907349e-6 from operation ordering.
- One original CHM scan/batch, nine batches/pass; eighteen finite passes total, including two B2 common-mask passes. All parameters frozen, no optimizer, no new training or masks, source artifacts opened read-only and rehashed unchanged afterward.
- PhysicalGPU3 UUID GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a; peak allocated CUDA memory369.974609MiB, diagnostic loop107.752842seconds (excludes some preflight/loading). Process exited after completion; no other jobs altered.
- Full task per-rate on/off/half metrics, class recalls/support, source hashes, checkpoint references, probability distributions and fixed calibration bins: RESULTS.json.
- Fresh completion verification: 51 diagnostic/statistics/runner/module tests plus11 attachment tests passed (62 total); git diff --check passed. Torch1.13 reports the known optional torch-sparse/libcusparse import warning; this task does not use that extension. No inference failure was recorded.
- Raw generated report: RESULT.md. Large per-utterance prediction/relation NPZ files remain only in the independent remote runs_v1 directory.
- This result does not separate E-specific information from the rest of the residual MLP, infer causal historical usefulness, establish emotion shift detection, or demonstrate statistical significance across seeds.
