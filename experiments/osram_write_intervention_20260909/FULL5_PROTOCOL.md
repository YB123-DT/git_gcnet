# Five-seed frozen-checkpoint intervention

Locked before evaluation: IEMOCAPFour, fold 5, seeds 66/67/68/69/70;
rates 0.0/0.1/0.3/0.5/0.7 inherited from the pilot. These five rates are
not an eight-rate benchmark. Four modes, one already-trained checkpoint
per seed, no new checkpoint selection, no training and no tuning.

- Reference: existing causal block write unchanged.
- Fixed0.9: multiply EVERY proposed update by 0.9, including complete input
  and steps without eligible historical missing addresses.
- DynamicGlobal (`global`): same-state norm-matched global weakening from
  the pilot. If no historical missing address exists it is identity.
- Protected: same historical-address projection from the pilot, ridge 0.001.

Weights, evaluation masks, preprocessing, read-before-write and metrics stay
paired within seed/rate. No changes to OSRAM production architecture, loss,
JEPA, MMoE, optimizer, alpha, beta or original ridge update are introduced.

## Current observed write-fit

For each valid currently observed slot and head, record relative fit error
`||M k - v|| / (||v|| + 1e-8)` at:

1. `err_before`: after decay, before the current write;
2. `err_original_after`: hypothetical unmodified block write from this SAME
   current trajectory state;
3. `err_after`: actual intervened post-write memory.

`fit_gain = err_before - err_after` stays signed; positive means this write
improves current association fit. Missing/padding slots never enter these
statistics. `target_modality` denotes the currently observed write-fit slot,
not a missing JEPA target. All recording is no-gradient and does not feed back
into inference. Old historical retention continues to use the existing
missing-only probe. Post-write errors describe memory available to FUTURE
utterances, not current classification read quality.

Compare fits within each run, aggregate seeds equally; heads and utterances
are not independent experimental replications. Report DynamicGlobal minus
Fixed0.9 by seed and rate plus descriptive uncertainty; with N=5, do not
claim robust equivalence from nonsignificance or package a gate from one win.
No new adaptive mechanism will be implemented automatically after analysis.

## Artifacts and reuse

Store new runs under `full5/iemocap4_seed{seed}/`; preserve pilot artifacts.
Repeat the pilot seed66 evaluations only to collect newly required observed
write-fit, then check original three modes match prior task metrics exactly.
All 100 forwards (5 seeds × 5 rates × 4 modes) are evaluation-only.
Checkpoint paths: existing `osram_forward_only_iemocap_20260908/iemocap4/seed_*/best.pt`.
Historical Test-oracle checkpoint selection and IEMOCAP validation/test overlap
make this INTERNAL DIAGNOSTIC ONLY, NOT A FORMAL PAPER RESULT.

CPU processes, two PyTorch threads each; no GPU training or checkpoint writes.
Compressed raw records, checkpoint hashes, paired-mask hashes, per-rate task
metrics and per-head diagnostic summaries accompany the aggregate report.
