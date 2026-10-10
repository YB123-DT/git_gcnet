# Fixed Text contribution under changed history

INTERNAL DIAGNOSTIC ONLY; inherited per-rate Test-oracle checkpoints

Original Flat seed66. Background deletion exactly one A/V bit outside the focal Text utterance. Nearest eligible Text; closest other deletable bit, deterministic position/modality tie-breaks. No labels used for planning, no training. Four independent causal trajectories; target Local and focal input pairing checked. Delta=MSE(Text absent)-MSE(Text present). Positive means helpful.

| Rate | N | Delta A mean | Delta B mean | Mean abs J | Help→harm | Harm→help |
|---|---:|---:|---:|---:|---:|---:|
| 0.0 | 624 | 0.000103 | -0.001185 | 0.008568 | 3 | 5 |
| 0.1 | 621 | 0.019925 | 0.018555 | 0.010565 | 4 | 8 |
| 0.2 | 614 | 0.037831 | 0.036902 | 0.012488 | 2 | 5 |
| 0.3 | 601 | 0.058053 | 0.051862 | 0.015755 | 3 | 4 |
| 0.4 | 588 | 0.131754 | 0.129217 | 0.017606 | 3 | 4 |
| 0.5 | 549 | 0.020684 | 0.020606 | 0.009464 | 3 | 2 |
| 0.6 | 513 | -0.021685 | -0.019161 | 0.013465 | 1 | 1 |
| 0.7 | 421 | 0.034288 | 0.035040 | 0.009901 | 10 | 5 |

SUMMARY.json includes all four-condition eligible-subset W-F1, quantiles, fixed1e-6 sign deadband, rescue/harm counts and per-rate availability/deletion-modality/lag strata. The eligible subset is not the full test set. Repeated rates are not independent samples. Descriptive contribution reversals do not by themselves establish malfunction, a natural-language causal effect or an actionable new module. Single-seed, one background deletion strength; no threshold search.

## Completed paired interpretation

Completed all eight rates in 13min59s. 4,531 target/rate pairs, of which4,365
are nonneutral. These are repeated rate-specific observations, not4,531 unique
utterances. MSE helpful-to-harmful reversals29, harmful-to-helpful34: pooled
fraction1.3904%; equal-rate fraction1.4467%. Both counts remain unchanged using
the preregistered1e-6 deadband. No claim of statistically significant frequency.

Equal-rate mean Delta_A=.0351193, Delta_B=.0339796;
signed J=-.00113967, mean absolute J=.0122265. Background-dependent numerical
contribution exists, but under a single A/V-bit deletion contribution sign
reversal is uncommon in this construction. Rate.7 has15/421 reversals(3.563%).

Crucially, MSE sign reversal is NOT polarity correction reversal:
there are ZERO cases A_rescue AND B_harm, and ZERO cases A_harm AND B_rescue.
Thus these records do not support frequent changes from polarity rescue to harm.
The continuous sentiment score can become closer/farther without changing its
predicted polarity. All-neutral labels remain in MSE counts but are excluded
from polarity statistics, exactly as planned.

| Background | Text preserved: eligible-subset mean W-F1 (%) | Text deleted | Preservation increment (pp) | Rescue / harm occurrences |
|---|---:|---:|---:|---:|
| A: original history | 81.934 | 80.943 | +0.991 | 173 / 130 |
| B: one other A/V bit removed | 82.040 | 80.890 | +1.150 | 179 / 129 |

These are equal-eight-rate W-F1 means on identical eligible targets within each
rate, NOT the original whole-test mean81.068. Subset memberships differ across
rates; do not compare these81.934/82.040 scores with full-test baseline as model
improvements. The small descriptive background difference is not evidence that
deleting history generally improves prediction.

Verification: all4531 contribution formulas and polarity flags independently
recomputed; unique target IDs within each rate; frozen-state confirmations for
eight checkpoints. Current Local maximum difference0, focal Text-pair Local
difference0; original prediction replay maximum error7.15256e-7. Source code
hashes match the pre-run provenance. AUDIT.json retains raw JSON hashes and
coverage; paired_predictions.csv retains all four predictions and identities.
Raw records remain under biggpu
`/data1/yb/remote_experiments/osram_history_text_context_20261010/results`.

Conclusion: this bounded diagnostic finds background-dependent numerical effects,
but not frequent harmful decision reversals. It does not justify automatically
adding a context-conditioned filter, and does not rule out stronger or different
background interventions. No additional inference or training was performed for
the final audit.
