# Interpretation — frozen history diagnostic

This is seed66 original cfg84 Flat, not the paired-view trained model. No training,
new loss, gate, or backbone modification. All figures are internal diagnostics
using historically Test-oracle-selected checkpoints, not formal performance.

1. Isolation passed: same current raw features/availability exactly; Local differs
   only at the calibrated float32 numerical floor. The original eight test W-F1
   values were reproduced. Each full causal forward initializes its own memory.
2. History changes alter Base/active Gap, then hidden and predictions. Text-only
   history deletion has larger average prediction shifts than A-only/V-only in
   both validation and test. This ordering survives matching the same anchors
   across A/T/V; it is not solely due to comparing different current utterances.
3. Test rate-macro prediction shifts: A .00769, T .06894, V .01821, mixed .05943.
   Corresponding binary flip rates: .386%, 3.587%, .322%, 2.616%. Neutral labels
   excluded for classification; repeated exposures across rates are not independent.
4. Matched test anchors (1379 exposures each): A .00507, T .05702, V .01554;
   flip rates .156%, 2.998%, .102%. Deletion timing/counts still differ; this is
   not an equal-dose randomized estimate of modality importance.
5. Mixed-history test shifts decrease with nearest deletion distance: .11267
   (1 step), .06498 (2), .03392 (3–4), .01165 (5+). This is consistent with stronger
   short-range sensitivity, not proof of a causal decay law: trajectories can
   contain multiple deletions and differ in dose/current pattern.
6. Harm is not universal. Text deletion produces 52 correct→wrong and 40 wrong→correct
   test exposures; validation has 10 versus12. These are pooled counts, whereas
   headline rates are rate-macro means. This does not establish robust harmful
   drift, excessive amplification, or that all history differences should align.

The defensible finding is **current-input-controlled sensitivity to history Text
availability**, not “Memory is wrong.” Hidden cosine changes are small on average;
cross-layer cosine magnitudes are not an amplification factor. These measurements
do not explain the paired-view training degradation causally or validate a new block.
Full per-rate, current-pattern, individual Gap, distance and actual-prefix groups
are retained so future hypotheses need not be inferred from averages alone.
