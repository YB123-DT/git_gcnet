# Proposed MELD-only grouped Memory tuning

Status:PLANNED ONLY. No training or queued jobs launched by this proposal.
This follows the user's proposed grouped search; frozen diagnostic results do
not establish the optimal settings of independently trained models.

Follow-up:user requested launch on2026-10-17. First-round execution is tracked
in experiments/osram_meld_write_tuning_20261017. It retains the accepted original
per-rate Test-best ACC protocol as internal exploration,not a silent validation
switch or a formal optimal-setting claim. The planning-only statements below
describe the initial proposal;later rounds remain unlaunched/unauthorized.

## First round:five one-factor configurations,not a3x3grid

| ID | osram_write_step | osram_write_ridge | Difference from control |
|---|---:|---:|---|
|W0|0.6|0.001|Matched new training control|
|W1|0.3|0.001|Write correction step only|
|W2|0.9|0.001|Write correction step only|
|W3|0.6|0.0001|Joint-solve write ridge only|
|W4|0.6|0.01|Joint-solve write ridge only|

Shared reference:accepted original ECERC MELD Flat1600 with Local Skip,
seed66,50epochs,batch32,Adam lr0.001,weight_decay0.00001,dropout0.5,
cyclic gcnet-random-mask-v1,no-JEPA/emotion-only,8heads,K/V64,
causal read-before-write and one joint observed-A/T/V Block Delta Write.
Keep query read_ridge unchanged:write_ridge and read_ridge are distinct.
No Block2,Nested,Gate,completion,paired views,new loss or dimension search.
Use the original MELD feature SHA256 recorded in BASELINE.md and cmerc-repro
on biggpu;GPU4remains forbidden. No server/environment migration.

When explicitly executed,these are independent from-scratch training runs,
not inference overrides or loading BEST weights as new initialization. The
new W0 must share the same pinned numerical source,initial weights,RNG/mask
stream,environment and optimizer-update budget as W1–W4. The accepted historic
seed66baseline is a provenance/sanity reference,not a substitute for this
matched control:byte-for-byte fresh-training reproduction has not been shown.
Save exact selected-epoch masks,predictions,ACC/UA/W-F1/CE/class recalls and
checkpoints together;UA must not come from a different epoch/random re-test.
Verify the shared mask stream rather than assuming equal seeds guarantee it.

The accepted baseline used per-rate Test-oracle ACC selection. That historical
protocol is not silently relabeled as validation selection. Configuration
ranking/final-model selection must be explicitly locked before launch;the user
has not specified it for this new training proposal. The latest diagnostic
instruction forbids using frozen test interventions as optimal parameter
selection. Do not silently launch a different selection protocol or conflate
exploratory Test-best scores with independent-test evidence.

## Second round:conditional,not automatically queued

Only if first-round task results do not show a clear useful response,consider:

- alpha_init0.95/0.98/0.995,with alpha still trainable. This is an initialization
  study,not fixing retention throughout training. Record actual final/selected
  per-head alpha values and their trajectories to detect convergence to the same
  solution. Actual code at ff81a8f exposes alpha_init only in OSRAMBackbone,
  not TrainConfig/MissingM3GraphModel;safe configuration plumbing and default-path
  regression tests would be needed before this round can run.
- gap_residual_strength0/0.5/1.0 with osram_gap_read=residual. Zero means the
  original Gap query is read without residualization,NOT Gap off. Existing
  strength1.0is the control. This changes addressing,not the joint write rule.

If this round is reached,hold the first-round write settings fixed at a
predeclared reference and change one factor at a time. Avoid re-running a
duplicate control without a version/protocol reason. Defaults are not claimed
optimal. Do not combine both rounds into one large grid.

## Third round:limited joint search only after evidence

Retain only parameters with reproducible task signal rather than improved
source-key reconstruction or changed internal norms alone. Report ACC/UA
tradeoffs,CE and high-missing behavior;do not rank on whichever metric improved
after looking. Fix a finite search budget,selection rule and promoted settings
before launch. Do not add heads/key/value sizes or launch Optuna automatically.
There is no universal numeric threshold for a useful effect registered here;
a one-example Test-best change from the frozen audit is not such evidence.

## Existing evidence and deduplication

The completed functional audit changed alpha/beta at inference,not write_step
or write_ridge during training. Existing write-retention forks remove decay or
intervening writes;they also do not replace this proposed trained comparison.
MELD experiment documentation and relevant runners were searched locally;no
completed matched W0–W4 training sweep was found. This is a bounded repository
lookup,not proof that no unindexed remote run exists. Check live/archived remote
jobs before any eventual launch. No completed experiment is re-run now.
