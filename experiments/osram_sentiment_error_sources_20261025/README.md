# Frozen Flat error-source accounting

INTERNAL DIAGNOSTIC ONLY. No new training, inference, mask sampling or GPU use.

The user requested all eight missing rates split by current missing modalities,
then by whether each missing modality had ever been observed in the strictly
preceding conversation. This directory provides the exact joint split as well
as overlapping per-target marginals. It is not an attribution of error causes.

## Fixed sources

Both are completed original cfg84 LargeFlat seed66 in **cmerc-repro**, not the
interrupted s0 Conditional Reread jobs and not Flat128/N10:

| Dataset | Eight-rate ACC | Eight-rate W-F1 | Source |
|---|---:|---:|---|
| MOSI |79.535061|79.460447|`/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2`|
| MOSEI |83.492708|83.288796|`/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSEI/seed_66/flat`|

MOSI150epochs/LR3e-4 then3e-5; MOSEI60epochs/LR1e-4. Original GCNet random
mask v1, batch32/Adam/WD1e-5/MSE, saved per-rate Test-best W-F1 checkpoints.
Each rate may select a different epoch. All existing predictions/masks/weights
and other running tasks are untouched; this analysis reads saved NPZ files only.

## Outputs and interpretation

- `results/RESULT.md`:all rate/current-set errors, history marginals, then full
  rate x current-set x joint history states, with errors/N and error rates.
- `results/CURRENT_PATTERNS.csv`:128rows, including ALL for each dataset/rate.
- `results/JOINT_HISTORY.csv`:304mutually-exclusive joint-state rows.
- `results/MISSING_MODALITY_HISTORY.csv`:144overlapping modality/state rows,
  including never_with_history as a first-turn-excluded subset of never.
- `results/SUMMARY.json`:all statistics, exact source environments, selected
  epochs and prediction/mask/checkpoint hash references. No raw features or
  prediction arrays are committed.
- `results/VERIFICATION.json`:independent per-record Counter verification.

Errors exclude gold0 and compare prediction>0 with gold>0. A prediction of0
is negative. History uses availability of **all valid preceding turns**,
including gold0 turns. No current or future availability enters seen flags.
The first utterance is never-seen for every modality; its counts are explicit.
Seen denotes prior observation, not successful Memory storage or retrieval.

For current A, missing targets are T/V. Joint states are T-never/V-never,
T-never/V-seen, T-seen/V-never and T-seen/V-seen. They partition the A group.
The per-target marginal tables separately pool current A/V/AV for missing T,
for example. An A-only record belongs to both missing-T and missing-V marginals,
so these columns must not be summed to recover the overall error count.

## A concrete descriptive finding

At rate0.7, MOSI has196errors, of which166are current-no-Text;144of those166
already have past observed Text. MOSEI has788errors, of which616are current-no-Text;
376of those616already have past observed Text. Error rates for missing-T
never/seen groups are40.0%/36.7% in MOSI and28.95%/23.24% in MOSEI.

Thus lack of any past observed Text is not a sufficient description of most
current-no-Text errors. These counts do NOT show that the Memory lost the
information, that a classifier failed to use it, or that past Text necessarily
contains enough information to classify the current utterance. Supports,
position, labels and content differ across groups; no causal conclusion follows.

## Checks and reproducibility

Four synthetic tests were observed failing before implementation, then passed:
strict past and conversation isolation under unsorted rows, duplicate/gap
rejection, zero-label/zero-prediction handling, neutral-history inclusion and
partition reconciliation. No large generic smoke suite was added.

All16 source prediction SHA256 match original manifests; independently
recomputed ACC/W-F1 match the recorded selected epochs. Current-set sums and
joint-state sums reconcile to every rate total. An independent Python Counter
implementation, without importing the analysis code, rechecked448joint/marginal
rows directly against saved predictions and confirmed first-turn counts.

```bash
CUDA_VISIBLE_DEVICES='' /data2/yb/conda_envs/cmerc-repro/bin/python \
  /data2/yb/remote_experiments/osram_sentiment_error_sources_20261025/analyze.py \
  --source-commit ca65ffd \
  --output /data2/yb/remote_experiments/osram_sentiment_error_sources_20261025/output_v1
```

This is the completed command; output_v1 now exists, and overwrites are refused.
Use a fresh output directory for a deliberate rerun. The parent Git revision
and exact analysis.py SHA256 are recorded in SUMMARY.json; this task's final
commit contains that script and the verified small reports. No model code changed.

Local test command:
`python -m unittest experiments.osram_sentiment_error_sources_20261025.test_analyze -v`.

Remote output directory:
`/data2/yb/remote_experiments/osram_sentiment_error_sources_20261025/output_v1`.
