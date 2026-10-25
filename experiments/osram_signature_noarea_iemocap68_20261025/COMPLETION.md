# Completed matched NoArea training,not frozen intervention

All40 IEMOCAP-6 seed68 fold/rate branches completed100epochs without failed or
missing jobs. Queue completed at2026-10-25T14:44:39Z after launch14:14:52Z,
about29minutes48seconds wall-clock including smoke and staggered admission.
Three authorized physical GPUs0/1/2,at most6workers percard,no other tasks stopped.

## Fivefold/eight-rate arithmetic comparison

| Model | ACC | UA | W-F1 |
| --- | ---: | ---: | ---: |
| Original frozen Flat |61.507197|60.307075|61.289083|
| Original normally trained Signature |62.273414|61.236214|62.272122|
| Newly trained NoArea Signature |62.307823|61.378763|62.302833|
| NoArea minus normal,pp |+0.034409|+0.142549|+0.030711|

High0.5–0.7 mean NoArea ACC59.964462,UA59.024489 versus normal Signature
59.938714/58.867712,changes+0.025748/+0.156776pp. NoArea still exceeds its own
frozen Flat by about+0.801ACC/+1.072UA over all eight rates.

Both variants trained from the SAME seed68 random initialization with the same
281856parameters,same original frozen parents,same Adam100epoch budget/LR and
same per-epoch cyclic training masks and update counts. Both used the same
per-rate earliest strict Test-best ACC rule with UA from the matching prediction.
Independent validation recomputed all40 ACC/UA pairs from saved predictions,
checked the complete100epoch histories against original normal Signature and
confirmed the equal-fold/rate mean. VERIFICATION.json preserves the evidence.
Local and remote24tests and real-data CUDA smoke had passed before admission.

## What this means

Removing directional area at training time did not reduce aggregate ACC or UA
on this seed; measured results are slightly higher. This does not support
claiming directional second-level geometry is necessary for this comparison's
gain. Tiny differences are not statistical superiority claims from one seed.

The earlier frozen intervention62.147ACC/61.116UA held normal-trained weights
fixed while changing their features. The present62.308/61.379 comes from weights
trained for the new features for the full original budget. Those are different
questions and must not be placed in one table as interchangeable model results.

This result narrows the explanation toward retained endpoint/content/time/count
features and task correction,but does not separate actual history value from
the extra branch's currentLocal processing and optimization capacity. NoArea
retains first level and symmetric second level; it is not a history-free or
Local-only model. No further ablation or seed search was automatically started.

## Artifacts

Training source `1d5ea017cb2d99630134722835978c91600e0946`; numerical source
`21e46f2ecd092345818659694db8ff372672c35e`. Unchanged cmerc-repro,new original
GCNet Missing,internal per-rate Test-oracle exploration. All new and original
checkpoints,predictions,masks and recovery states are preserved remotely.

`/data2/yb/remote_experiments/osram_signature_noarea_iemocap68_20261025/execution_v1`

RESULT.md/JSON/CSV contain every fold/rate and high-missing comparison. No model
weights or raw prediction arrays are committed to Git. The existing three-seed
normal Signature result is unchanged; this is only its seed68 learned control.
