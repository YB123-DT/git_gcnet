# High-scoring existing MOSI candidates

INTERNAL DIAGNOSTIC ONLY. Read-only review; no new training or inference.
Scope: one-stage cfg84 module candidates, seed66/100epochs/per-rate BEST.
This is a shortlist, not an exhaustive ranking of every historical run or
every hyperparameter variant. No frozen two-stage or oracle splice results.
Eight rates0.0–0.7; high=.5/.6/.7. Units percent.

|Candidate|ACC mean8|W-F1 mean8|W-F1 high|Integration|
|---|---:|---:|---:|---|
|Flat reference|81.174|81.068|76.352|Unmodified reference|
|XCiT-XCA|81.117|81.042|76.257|Zero-init residual into Flat pre-norm anchor|
|Nested GNN|81.059|80.992|76.077|Residual into adapter Local/Base/Gap; original Local Skip|
|Neural Production|81.021|80.981|76.475|Residual into Base/Gap only; Local unchanged|
|D3-W256 scalar shift filter|80.926|80.932|76.200|Filtered relation-space shift projected as Flat residual|
|CWN cellular evidence|80.812|80.821|76.004|TokenAdapter residual into adapter Local/Base/Gap|
|Perceiver IO|80.983|80.818|76.126|Flat pre-norm residual|
|Tucker fusion|80.983|80.813|75.984|Flat pre-norm residual|
|Sheaf hypergraph|80.888|80.776|76.236|Flat pre-norm residual|

## Completed three-seed confirmation, seeds66/67/68

|Method|W-F1 mean8|W-F1 high|
|---|---:|---:|
|Flat|80.559|75.594|
|Nested|80.489|75.252|
|Neural Production|80.459|75.558|
|XCiT-XCA|80.213|75.365|
|D3-W256|79.953|74.992|

Nested is closest to Flat mean8 among these confirmed module candidates;
Neural Production is closest on high missing. Neither establishes improvement.
CWN, Perceiver, Tucker and Sheaf have seed66 evidence here, not a verified
three-seed confirmation. R12 continuous-label contrastive training separately
scores80.843/75.752 on seed66 and80.038/75.125 on3seeds; it is a training objective,
not a new inference block, so omitted from the block shortlist above.

## Sources checked

- Remote priority40 group1/group2 runs: m28_xcit_xca, m09_tucker; metrics.json and100-epoch histories.
- Remote osram_new40_gpu0123_20261004/attempt2/runs: nested_gnn_rooted_evidence,
  conditional_new_07_neural_production, cwn_cellular_evidence; seed66 metrics/history.
- Remote osram_meaningful20_20261003_round1/runs/{perceiver_io,sheaf_hypergnn_diag}/seed_66/metrics.json.
- experiments/osram_meaningful20_20261003/SUMMARY.json and RESULT.md.
- experiments/osram_shift_capacity_20261002/results/D3-W256/metrics.json,
  RESULT.md and THREE_SEED_RESULT.md.
- experiments/osram_readout_top3_3seed_20261005/RESULT.md and SUMMARY.json.
- experiments/osram_r12_3seed_20261005/RESULT.md.
- Integration: MeaningfulReadoutResidual, TokenAdapter, conditional.TokenReadout,
  MemoryShiftFilter and priority40 registry. Names alone were not used to infer placement.

Review implication: these stronger examples span both Flat-output residuals
and evidence-input residuals. Scores cannot be generalized to arbitrary direct
replacement implementations or their original source-paper architectures.
