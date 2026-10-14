# Same-config repeat warning

INTERNAL TEST-ORACLE SCREEN. This is a seed66 repeat check supplied by the
pre-registered default Optuna trial, not a new seed or additional experiment.

The B6 screen and `hpo_B6_000` use the same immutable training source
`7abb85567ee8f81c042d59d17c55b4ea501df4dc`, seed66, 50 epochs, feature file,
GCNet random-mask protocol, and common-mask replay. Their effective configuration
dictionaries are identical after excluding the output directory. Both runs
completed successfully. Nonetheless, the paired results differ:

| Run | ACC | UA | W-F1 | CE |
| --- | ---: | ---: | ---: | ---: |
| B6 screen | 60.407088 | 36.914617 | 57.175042 | 1.232490 |
| B6 default trial 0 | 60.550766 | 34.624924 | 56.610239 | 1.196578 |

The UA difference is -2.289692 pp. Consequently, the screen's UA increase is
not evidence of a stable improvement. No precise numerical/root-cause attribution
has been established, and this difference must not be hidden or reclassified as
a failed trial. The default training implementation does not enforce strict
deterministic PyTorch algorithms. That is a limitation, not a demonstrated
explanation for the observed discrepancy.

The initial screen ranking remains the pre-registered ranking: B6 and B2 advance.
The finite HPO budget continues without changing the source, selection rule, or
mask protocol. Final HPO reporting must retain this repeat alongside all trials,
and any later multi-seed recommendation must account for this instability and
the very small same-checkpoint B6-on versus B6-off contribution.

Evidence is retained remotely under
`/data2/yb/remote_experiments/osram_meld_block2_auto_20261014_v2/` in
`screen/B6/SUMMARY.json` and `optuna/runs/hpo_B6_000/SUMMARY.json`.
