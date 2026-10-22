# Joint KV Fit and Future Task Utility

INTERNAL DIAGNOSTIC ONLY

No new training. Same protected frozen Flat checkpoint and saved GCNet masks.

Q = normalized KV error before write minus after write; U = future MSE(omit) - MSE(keep).
Omit preserves decay and skips only the selected joint write; future writes resume normally.

| Group | Condition | Pair slots | Conversations | Mean Q | Mean U | Spearman | Helps | Harms | Neutral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | all | 18232 | 31 | 0.370570 | 0.000857 | -0.016252 | 9007 | 9225 | 0 |
| by_rate | 0.0 | 2279 | 31 | 0.331114 | -0.001959 | -0.085656 | 1131 | 1148 | 0 |
| by_rate | 0.1 | 2279 | 31 | 0.336578 | 0.001890 | -0.053041 | 1148 | 1131 | 0 |
| by_rate | 0.2 | 2279 | 31 | 0.340870 | 0.001249 | -0.046922 | 1124 | 1155 | 0 |
| by_rate | 0.3 | 2279 | 31 | 0.359019 | 0.004430 | -0.016240 | 1148 | 1131 | 0 |
| by_rate | 0.4 | 2279 | 31 | 0.364209 | -0.004328 | -0.046652 | 1078 | 1201 | 0 |
| by_rate | 0.5 | 2279 | 31 | 0.382641 | 0.009820 | 0.017479 | 1137 | 1142 | 0 |
| by_rate | 0.6 | 2279 | 31 | 0.418092 | 0.010704 | 0.005210 | 1135 | 1144 | 0 |
| by_rate | 0.7 | 2279 | 31 | 0.432036 | -0.014954 | 0.019379 | 1106 | 1173 | 0 |
| by_distance | 1 | 5240 | 31 | 0.362808 | 0.005263 | -0.014028 | 2515 | 2725 | 0 |
| by_distance | 2 | 4992 | 31 | 0.365680 | -0.004090 | -0.039662 | 2506 | 2486 | 0 |
| by_distance | 4 | 4496 | 31 | 0.369844 | 0.002657 | -0.000835 | 2267 | 2229 | 0 |
| by_distance | 8 | 3504 | 31 | 0.390075 | -0.000996 | -0.005407 | 1719 | 1785 | 0 |
| by_target_pattern | A | 2334 | 31 | 0.419812 | -0.007183 | -0.002171 | 1160 | 1174 | 0 |
| by_target_pattern | AT | 1550 | 31 | 0.372673 | 0.003990 | 0.007218 | 774 | 776 | 0 |
| by_target_pattern | ATV | 6250 | 31 | 0.334568 | -0.002617 | -0.079821 | 3063 | 3187 | 0 |
| by_target_pattern | AV | 1793 | 31 | 0.358937 | 0.010372 | 0.002707 | 945 | 848 | 0 |
| by_target_pattern | T | 2285 | 31 | 0.394772 | 0.002831 | 0.022387 | 1146 | 1139 | 0 |
| by_target_pattern | TV | 1730 | 31 | 0.364586 | -0.002772 | -0.081977 | 800 | 930 | 0 |
| by_target_pattern | V | 2290 | 31 | 0.406696 | 0.009729 | 0.030026 | 1119 | 1171 | 0 |
| by_distance_target_pattern | 1|A | 672 | 31 | 0.411302 | 0.000875 | -0.005244 | 335 | 337 | 0 |
| by_distance_target_pattern | 1|AT | 445 | 31 | 0.368383 | 0.007951 | 0.006085 | 218 | 227 | 0 |
| by_distance_target_pattern | 1|ATV | 1788 | 31 | 0.329697 | -0.006625 | -0.105118 | 820 | 968 | 0 |
| by_distance_target_pattern | 1|AV | 511 | 31 | 0.340981 | 0.035013 | 0.067950 | 265 | 246 | 0 |
| by_distance_target_pattern | 1|T | 655 | 31 | 0.386890 | 0.003436 | 0.044761 | 319 | 336 | 0 |
| by_distance_target_pattern | 1|TV | 504 | 31 | 0.365756 | -0.002635 | -0.073119 | 235 | 269 | 0 |
| by_distance_target_pattern | 1|V | 665 | 31 | 0.389917 | 0.024785 | 0.039432 | 323 | 342 | 0 |
| by_distance_target_pattern | 2|A | 636 | 31 | 0.405787 | -0.045695 | -0.095159 | 309 | 327 | 0 |
| by_distance_target_pattern | 2|AT | 421 | 31 | 0.377541 | 0.008989 | 0.036436 | 223 | 198 | 0 |
| by_distance_target_pattern | 2|ATV | 1708 | 31 | 0.332201 | -0.001606 | -0.062150 | 875 | 833 | 0 |
| by_distance_target_pattern | 2|AV | 486 | 31 | 0.342655 | 0.000075 | -0.058867 | 264 | 222 | 0 |
| by_distance_target_pattern | 2|T | 626 | 31 | 0.372552 | 0.007050 | 0.062514 | 324 | 302 | 0 |
| by_distance_target_pattern | 2|TV | 481 | 31 | 0.373479 | -0.005711 | -0.156465 | 224 | 257 | 0 |
| by_distance_target_pattern | 2|V | 634 | 31 | 0.412713 | 0.009308 | -0.000730 | 287 | 347 | 0 |
| by_distance_target_pattern | 4|A | 581 | 31 | 0.414405 | 0.023098 | 0.092609 | 311 | 270 | 0 |
| by_distance_target_pattern | 4|AT | 380 | 31 | 0.352110 | -0.001346 | -0.014431 | 190 | 190 | 0 |
| by_distance_target_pattern | 4|ATV | 1543 | 31 | 0.334588 | -0.000935 | -0.098061 | 773 | 770 | 0 |
| by_distance_target_pattern | 4|AV | 448 | 31 | 0.379839 | 0.001472 | -0.011337 | 223 | 225 | 0 |
| by_distance_target_pattern | 4|T | 562 | 31 | 0.404123 | 0.000954 | -0.012150 | 297 | 265 | 0 |
| by_distance_target_pattern | 4|TV | 423 | 31 | 0.351216 | -0.000938 | 0.024133 | 193 | 230 | 0 |
| by_distance_target_pattern | 4|V | 559 | 31 | 0.404525 | -0.000568 | 0.034186 | 280 | 279 | 0 |
| by_distance_target_pattern | 8|A | 445 | 31 | 0.459768 | -0.003844 | 0.057148 | 205 | 240 | 0 |
| by_distance_target_pattern | 8|AT | 304 | 30 | 0.397917 | -0.002062 | -0.002903 | 143 | 161 | 0 |
| by_distance_target_pattern | 8|ATV | 1211 | 31 | 0.345072 | -0.000267 | -0.115716 | 595 | 616 | 0 |
| by_distance_target_pattern | 8|AV | 348 | 31 | 0.381133 | 0.000026 | 0.100224 | 193 | 155 | 0 |
| by_distance_target_pattern | 8|T | 442 | 31 | 0.426033 | -0.001654 | -0.029999 | 206 | 236 | 0 |
| by_distance_target_pattern | 8|TV | 322 | 31 | 0.367033 | -0.001004 | -0.081973 | 148 | 174 | 0 |
| by_distance_target_pattern | 8|V | 432 | 31 | 0.426506 | 0.000497 | 0.073070 | 229 | 203 | 0 |
| q_median_split | low_or_equal | 9118 | 31 | 0.154119 | 0.001866 | 0.008542 | 4574 | 4544 | 0 |
| q_median_split | high | 9114 | 31 | 0.587115 | -0.000153 | -0.031153 | 4433 | 4681 | 0 |
| q_quartiles | Q1 | 4561 | 31 | 0.097835 | -0.000457 | 0.018467 | 2256 | 2305 | 0 |
| q_quartiles | Q2 | 4557 | 31 | 0.210453 | 0.004190 | -0.002781 | 2318 | 2239 | 0 |
| q_quartiles | Q3 | 4558 | 31 | 0.367877 | 0.005577 | -0.033033 | 2263 | 2295 | 0 |
| q_quartiles | Q4 | 4556 | 31 | 0.806450 | -0.005886 | -0.013043 | 2170 | 2386 | 0 |

## Conversation-cluster uncertainty

{"clusters": 31, "replicates": 200, "seed": 66, "mean_utility_ci95": [-0.005291413562437557, 0.009107801800511243], "spearman_ci95": [-0.042046110643154055, 0.010223201082147623], "valid_spearman_replicates": 200}

## Limits

- INTERNAL DIAGNOSTIC ONLY: one frozen, previously Test-oracle-selected checkpoint; no independent generalization claim.
- No new training, no parameter selection, and no change to GCNet Missing or original write equations.
- Utility measures a joint coupled write intervention plus all subsequent state evolution; it is not semantic source attribution.
- Repeated writes, targets, rates and distances are not independent samples; no iid p-values are reported.
- KV reconstruction quality is an internal association objective, not a direct measure of emotion relevance.
- Correlation does not establish mechanism or causality beyond the specified frozen-model write intervention.
- Median/quartile boundaries describe the observed data; they are not deployable thresholds or optimized settings.
- Within-quartile correlations are descriptive range-restricted associations, not independent confirmatory tests.
- MSE utility includes all valid labeled targets, including neutral labels; label=0 is excluded only from polarity counts.
- Conversation-cluster bootstrap intervals describe this sampled audit, not checkpoint/seed uncertainty.
