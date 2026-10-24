# MOSEI Text-history cohorts: frozen Local/Base/Gap readout replay

INTERNAL DIAGNOSTIC ONLY. No training or new mask sampling.

## Scope and provenance

Original repaired-RNG cmerc-repro Flat1600 seed66, miss0.7 selected epoch23.
LR1e-4 constant/60epoch training; gcnet-random-mask-v1. One unchanged
causal Memory scan per batch, then seven original Flat readouts with captured
Local/Base/Gap groups retained or zeroed. No query, write, current feature,
Local encoding, parameter, or availability intervention.

Reference run:
`/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none`.

Numerical source:
`/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/source_v2`,
original run source Git SHA `3777ab1a9210eecf83ffa5c47fb0ba87a37edae5`.
All original model/sampler Python file hashes were verified against the run.
Checkpoint SHA256:
`b8295472d37c451e29a9ab1c7b62ae6d7ee41ad469d7192ad2f3606592d0b4d2`.
Mask SHA256:
`a48c09ddf213e68dc563c7c5df1b8e2a7727d3185f53fd3afaf3391d00b79562`.
Selected predictions SHA256:
`70c43fe154e5f67f7e5beb47a3c9c5c0931b0bb399e37d4fa433e9976549be94`.

## Results (%)

Current Text is missing in both cohorts. History exposure is calculated
strictly before the current utterance, independently within each conversation.
All/nonneutral counts are 2036/1618 for history-seen-T and
1119/829 for history-never-T. Neutral labels are excluded from polarity ACC/W-F1,
but included in MSE. Positive means scalar prediction strictly greater than0.

| Readout inputs | Seen T: ACC | Seen T: W-F1 | Never T: ACC | Never T: W-F1 |
|---|---:|---:|---:|---:|
| Local | 60.321 | 51.152 | 72.618 | 66.069 |
| Base | 41.595 | 39.546 | 33.414 | 27.856 |
| Gap | 34.858 | 34.212 | 35.344 | 33.614 |
| Local+Base | 72.064 | 70.761 | 72.738 | 68.859 |
| Local+Gap | 75.464 | 75.036 | 73.703 | 68.829 |
| Base+Gap | 40.482 | 37.287 | 33.534 | 27.684 |
| Local+Base+Gap | 75.464 | 75.301 | 72.497 | 69.654 |

## Paired polarity corrections and harms

| Comparison | Seen T corrections/harms | Never T corrections/harms |
|---|---:|---:|
| Local -> Local+Base | 272/82 | 25/24 |
| Local -> Local+Gap | 365/120 | 19/10 |
| Local -> Full | 395/150 | 37/38 |
| Local+Base -> Full | 127/72 | 12/14 |

Full minus Local: seen-T ACC+15.142pp, W-F1+24.148pp;
never-T ACC-0.121pp, W-F1+3.585pp. In the never-T cohort, a higher W-F1
does NOT mean more correct predictions: full has one fewer correct polarity.
The class-specific distribution of errors matters for W-F1.

Seen-T Local+Gap and Full have equal ACC but are not identical predictions:
adding Base corrects33 and harms33. Full W-F1 is0.264pp higher and
MSE0.949457 vs Local+Gap0.974859.

## Interpretation limits

Memory reads substantially affect the existing full model in the seen-T
cohort. The overall high-missing score does not justify claiming Memory is
uniformly ineffective. Conversely, this is not a source-specific Text deletion:
all Base/Gap evidence is removed, not only Text-derived contributions. Cohorts
also differ in current samples, label balance, and available history; their
between-cohort differences do not establish a causal role of Text exposure or
conversation length.

Local/Base/Gap-only scores are inference-time interventions on ONE full-model
checkpoint, NOT separately trained classifiers or representation probes.
Zeroing inputs can be out of distribution. Poor Base/Gap-only performance
does not establish poor historical representation quality.
Never-seen-Text does not mean no history: Audio/Visual history may exist.

## Verification and execution

- Original complete full-cohort metrics reproduced: ACC78.343423,
  W-F1 78.034702, MSE0.761608.
- Original saved polarity predictions exactly reproduced.
- Maximum absolute scalar difference across saved/full replay:1.1920929e-7.
- 22 test batches, exactly 22 actual `_scan` calls; replay never calls `_scan`.
- Strict checkpoint loading; frozen model parameter digest unchanged.
- Exact saved labels, conversation IDs, utterance indices, speakers and masks.
- Inactive Gap values and causal backward halves verified zero.
- 3 new cohort/count tests plus 5 existing Flat source-selection/replay tests:
  8 passed under remote cmerc-repro;3 new tests passed locally.
- GPU7 `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`; no foreign process changed.
- Wall time42.159seconds including setup; peak allocated GPU350.124MiB.
- Existing optional torch-sparse import warning appeared; the original non-sparse
  OSRAM path ran to completion and matched the accepted predictions.

The first wrapper attempt stopped BEFORE model setup because the archived
worker's `setup` does not accept newer seed/fold kwargs. The standalone wrapper
was corrected to its real API, with explicit seed66/fold1 validation, then rerun
in fresh `execution_v2`. `execution_v1` and `replay_v1.log` remain preserved;
no old checkpoint or numerical source was changed. Local default Python lacks
Torch, so reused Torch tests were run in the actual remote cmerc environment.

Remote output:
`/data2/yb/remote_experiments/osram_mosei_text_history_readout_20261024/execution_v2`.
`RESULT.json` includes full metrics/config/hashes/checks; `predictions.npz`
remains remote and is not committed. `replay_v2.log` records each batch.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python evaluate.py \
  --source /data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/source_v2 \
  --reference /data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none \
  --output /data2/yb/remote_experiments/osram_mosei_text_history_readout_20261024/fresh_replay \
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

The standalone wrapper reuses existing source capture/selection/replay helpers;
no model refactor or new dependencies. Single-seed exploratory test-selected
diagnosis only; no new performance or causal mechanism claim.
