# Paired Local-adapter / Local-Skip / Memory diagnostic

User-specified design, no new gate, training, weight update or deployment
coefficient selection. For original Flat:

`hidden = LN(beta * S(Local) + Adapter([alpha * Local, mu * masked_memory]))`

13 fixed settings:9 with alpha=beta in{.8,1,1.2} crossed with mu in{.8,1,1.2};
plus(.8,1,1),(1.2,1,1),(1,.8,1),(1,1.2,1). Identity(1,1,1) included.
Beta scales the full Skip output including bias, not the Skip input.

## Scope and split audit

Run MOSI original cfg84 Flat seeds66/67/68, eight rates, train and validation
separately. MOSI data have52train conversations/1284utterances,10validation/229,
31test/686. Conversation sets disjoint. Test loader is never evaluated.
Train results are in-sample; validation conversations were held out from training.
Checkpoint provenance remains historical per-rate **Test-oracle**, so this is
NOT a fully test-independent validation-selected scientific result.
Every13-way comparison uses the identical checkpoint and identical inputs.
Checkpoint epochs may differ across rates, as in the established baseline.

IEMOCAP6 existing matched cfg84 official protocol aliases validation=test in all
15seed/fold configurations. Thus no IEMOCAP validation inference is authorized
by this non-test request. Cannot retrospectively label training conversations
as held-out. IEMOCAP deferred pending appropriate checkpoint or user choice of
explicitly in-sample analysis. CE/classification utility is unit-tested only;
no IEMOCAP results claimed.

## Implementation and acceptance

Hook original adapter input and complete Skip output during one eval forward.
Cache masked Base/Gap and Local; rerun only adapter/norm/original task head for13
settings. Do not change Memory, Query, mask schedules or original model files.
Assert identity logits bitwise equal; hash model state before/after checkpoint
diagnostics. Save checkpoint hashes, source hashes, ID splits, ordered utterance
IDs, labels, availability, original readout cache hashes and all13 predictions.
No optimizer exists. No coefficient is fitted or chosen.

Unit tests cover exact formulas, Skip bias, one-forward cache, padding/masks,
hook cleanup, no model-state change, MOSI neutral-label handling and CE stability.
Run a single validation cell smoke before full3seeds x8rates x2splits x13.
Expected full output48prediction files and624paired setting cells.

## Analysis

MOSI task loss is original MSE over all valid utterances (including neutral).
Report sample-weighted MSE, MAE/correlation and established nonzero-label >0
sign metrics. Corrections/harm use the same nonzero filter and threshold.
Loss means are not the old unweighted mean of batch means: explicitly aggregate
all valid utterances to avoid dependence on evaluation batching.
Do not equate lower MSE or net corrections with higher weighted F1.

Report each seed/rate and A/T/V/AT/AV/TV/ATV availability subset. Availability
subsets here are within random-missing trajectories, NOT persistent-mask tests.
Rates macro-average within each seed, then seeds macro-average. Absent pattern
cells omitted and actual cell counts recorded. Corrections summed across cells
count repeated evaluations, not unique utterances.

Also report per-conversation responses. Descriptive2000-replicate bootstrap
averages each conversation's repeated seed/rate loss deltas first, then samples
conversations. Intervals are not multiple-comparison-corrected; no significance
claim or best-coefficient recommendation. Validation has only10conversations.

Server biggpu GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
Original reference root:
`/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full`.
Dataset unchanged:
`/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset`.
