# Existing-log overlap versus signed write damage

Inputs: 40 existing runs; 311688 retention records; 311688 complete finite overlap/damage pairs. Missing overlap: 0; missing damage: 0.

Descriptive only: no p-values, independence assumptions, causal claims, or justification for a protection mechanism. Negative write_damage means improvement; it is never clipped. Missing/non-finite overlap is excluded, never set to zero. Metadata dataset/rate and record counts are checked against each log. Checkpoint selection remains the metadata oracle protocol.

Pearson/Spearman are computed within each dataset × seed × rate run on head-level records. Spearman uses average ranks for ties. Residual Pearson removes joint modality × head × history-distance-bucket means. Residual rank Pearson first globally ranks both variables within each run, then demeans ranks within the same joint strata; it is a categorical-adjusted rank association, not ordinary Spearman of raw residuals. Distance buckets: 1, 2–3, 4–7, 8+.

Conversation-equal metrics average defined within-conversation correlations with one vote per conversation; constant/singleton groups yield NA, and defined/positive counts are exported. Between-conversation correlations use one (mean overlap, mean damage) pair per conversation and measure a different, ecological association. Runs share evaluation conversations across rates/seeds; run sign counts are descriptive, not independent replications.

Bucket CSV retains all five fixed buckets, including empty ones. Its n is the number of complete head-level pairs; conversation_equal_damage_mean averages conversation means only among conversations present in that bucket. Sparse or empty strata/buckets do not support a universal monotonic claim. Strata with n < 10 are counted per run; singleton strata contribute zero residual variation; zero-variance correlations are NA.

| Dataset | Metric | Positive / defined runs | Median | Range |
| --- | --- | --- | --- | --- |
| ALL | pearson | 40/40 | 0.51010 | 0.28828 to 0.68299 |
| ALL | spearman | 40/40 | 0.55431 | 0.34362 to 0.73831 |
| ALL | residual_pearson | 40/40 | 0.52654 | 0.16871 to 0.69748 |
| ALL | residual_rank_pearson | 40/40 | 0.57304 | 0.19885 to 0.76571 |
| ALL | conversation_equal_pearson_mean | 40/40 | 0.53639 | 0.33038 to 0.69558 |
| ALL | conversation_equal_spearman_mean | 40/40 | 0.56730 | 0.35699 to 0.72988 |
| ALL | between_conversation_pearson | 35/40 | 0.39182 | -0.12950 to 0.84159 |
| ALL | between_conversation_spearman | 35/40 | 0.37863 | -0.11452 to 0.84294 |
| CMUMOSI | pearson | 20/20 | 0.39209 | 0.28828 to 0.46382 |
| CMUMOSI | spearman | 20/20 | 0.47315 | 0.34362 to 0.58678 |
| CMUMOSI | residual_pearson | 20/20 | 0.35589 | 0.16871 to 0.45783 |
| CMUMOSI | residual_rank_pearson | 20/20 | 0.45060 | 0.19885 to 0.54255 |
| CMUMOSI | conversation_equal_pearson_mean | 20/20 | 0.41496 | 0.33038 to 0.50979 |
| CMUMOSI | conversation_equal_spearman_mean | 20/20 | 0.48198 | 0.35699 to 0.62042 |
| CMUMOSI | between_conversation_pearson | 15/20 | 0.12179 | -0.12950 to 0.62682 |
| CMUMOSI | between_conversation_spearman | 15/20 | 0.11287 | -0.11452 to 0.57339 |
| IEMOCAPFour | pearson | 20/20 | 0.62627 | 0.55637 to 0.68299 |
| IEMOCAPFour | spearman | 20/20 | 0.64860 | 0.54532 to 0.73831 |
| IEMOCAPFour | residual_pearson | 20/20 | 0.65085 | 0.59524 to 0.69748 |
| IEMOCAPFour | residual_rank_pearson | 20/20 | 0.70043 | 0.60353 to 0.76571 |
| IEMOCAPFour | conversation_equal_pearson_mean | 20/20 | 0.62317 | 0.56298 to 0.69558 |
| IEMOCAPFour | conversation_equal_spearman_mean | 20/20 | 0.64388 | 0.54792 to 0.72988 |
| IEMOCAPFour | between_conversation_pearson | 20/20 | 0.66379 | 0.25522 to 0.84159 |
| IEMOCAPFour | between_conversation_spearman | 20/20 | 0.66069 | 0.36734 to 0.84294 |

## Stability and exceptions

Within-conversation Spearman is positive in 1236/1238 defined conversation–run groups (not unique independent conversations). Among 2984 joint strata, 600 have n < 10; 346/2856 defined stratum correlations are negative. Among 200 buckets, 0 are empty and 2 have n < 10.

Nondecreasing damage across all five ordered overlap buckets (complete-bucket runs only): damage_mean: 38/40; damage_median: 38/40; conversation_equal_damage_mean: 39/40.

Overall and adjusted associations, including the conversation-equal within-conversation summaries, are directionally stable across these runs. Individual strata, bucket monotonicity, and especially MOSI between-conversation associations are not universal. This is evidence of association in these recorded diagnostics, not evidence that overlap causes damage or that a protection change will improve predictions.

Reproduce: `python experiments/osram_retention_20260909/analyze_overlap_trends.py --self-test` and `python experiments/osram_retention_20260909/analyze_overlap_trends.py`.
