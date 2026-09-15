# Frozen Teacher Text prediction and probe-transfer audit

Completed 2026-09-15. No new neural-model training, optimizer/EMA updates, Test
evaluation, checkpoint reselection or classification feedback.

## Protocol

Seeds 66–70; current frozen-Teacher Student `best_miss_0p5.pt` checkpoints selected
at epochs 38/66/29/47/80. These were previously selected using per-rate Test oracle;
this audit does not make their provenance independent of Test selection.

Extract original contextual MMoE **reg_predictions**, not contrastive outputs.
Natural validation masks at missing=.5; select current A-only, V-only and AV
utterances. Other conversation nodes retain their natural masks. Thus A→T means
current A-only plus allowed causal context, not a context-free Audio predictor.
AV→T uses the original mean of directional predictions, not a new fusion.

For each seed, fit one StandardScaler + Ridge(alpha=10) on the **entire train
set's real frozen-Teacher Text latent** (1,284 samples) and continuous sentiment.
Reuse that exact scaler/head for real and predicted Text on the same validation
cohort. Do not recalibrate or fit a head on predictions. Binary W-F1 excludes
zero labels; latent metrics retain all valid utterances.

Controls: train-Text mean prototype and eight within-pattern prediction shuffles,
using the same frozen probe. Quality metrics reuse the existing audit's eight
target shuffles, cohort-centered cosine/retrieval, centered effective rank and
mean channel standard deviation. Retrieval chance is 1/cohort size, not 1/256.

## Text probe transfer

Validation W-F1 percentages, unweighted five-seed mean ± sample SD. The real and
predicted columns are paired on identical samples within each seed/pattern.

| Current sources → Text | Cohort counts, seeds 66–70 | Real Text | Predicted Text | Shuffled prediction mean | Train prototype mean |
|---|---|---:|---:|---:|---:|
| A→T | 31/41/34/40/46 | 86.44 ± 5.30 | 46.88 ± 13.45 | 44.66 | 41.18 |
| V→T | 37/42/50/29/36 | 87.06 ± 2.94 | 55.44 ± 8.67 | 47.43 | 42.59 |
| AV→T | 34/29/37/29/32 | 83.90 ± 7.79 | 54.56 ± 10.69 | 51.00 | 39.90 |

Real Text beats predicted Text in 5/5 seeds for each of the three patterns.
Different real-Text means across rows arise from different cohorts, not different
probe training. The full-validation real-Text score reproduces the preceding
information audit exactly for all five seeds (overall mean 85.55%).

## Sample prediction quality

Five-seed mean; cosine/gap/ratio are unitless. Rank is measured within each small
cohort and cannot be interpreted as a full-population 256-dimensional rank.

| Sources | Raw cosine | Centered cosine | Real−shuffle cosine | Retrieval / chance (%) | Prediction / target rank | std ratio |
|---|---:|---:|---:|---:|---:|---:|
| A | .655 | .137 | .032 | 4.23 / 2.66 | 3.07 / 11.31 | .233 |
| V | .645 | .184 | .045 | 4.14 / 2.66 | 3.10 / 11.91 | .264 |
| AV | .701 | .162 | .028 | 4.47 / 3.13 | 2.87 / 10.73 | .268 |

## Bounded interpretation

The evidence supports **strong Text target utility but weak transfer through the
current predicted Text coordinates**. The prediction carries limited sample
variation, weak correspondence and much less sentiment readability through the
fixed target-space probe. It is not justified to call every prediction a constant:
mean real−shuffle gaps are positive and V→T has a visible probe advantage over
shuffling. These small cohorts do not establish statistical significance.

Low direct probe transfer can reflect scale/offset or coordinate mismatch as
well as lost sentiment information. The predictor std is only 23–27% of target
std, and the probe deliberately applies the target's train-only standardization.
Therefore this is not proof that *no* alternative decoder could recover sentiment
from predictions. No calibrated/refitted prediction probe was introduced.

This does not identify why prediction is weak: missing-source ambiguity,
optimization, objective weighting and predictor capacity remain possible. It
does not prove a gradient conflict or justify immediately modifying MMoE. No
gradient audit, new loss or architecture was added in this task.

Validation has only 29–50 samples per pattern per seed and was used to select
Stage1 Teachers. Seeds share the dataset; rows are not independent dataset trials.
The strongest supported finding is the paired real-versus-predicted Text probe
gap, not a claimed universal failure of latent prediction.

## Evidence and checks

Code: `text_transfer_audit.py`; raw JSON: `text_transfer_results/seed_66.json`
through `seed_70.json`, plus `summary.json`. Per-sample latent/score NPZs remain
under remote `osram_supervised_teacher_20260914/text_transfer_audit`.

- Same checkpoint and frozen Teacher hashes verified for each seed.
- Predictor-on/off classification logits exactly identical.
- Whole-model state hashes unchanged; all model parameters frozen in no-grad.
- Train/validation conversation IDs disjoint.
- Full-validation real Text probe exactly matches the preceding audit.
- Four probe tests pass; compilation and `git diff --check` pass.
- Only analysis code/tests/artifacts changed; model, loss and inference unchanged.
