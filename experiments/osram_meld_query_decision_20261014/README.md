# Frozen Query versus Decision diagnostic

INTERNAL DIAGNOSTIC ONLY. MELD seed66, eight original large-Flat checkpoints.
This is training small diagnostic heads, not retraining CHM, and not pure inference.

## Controlled comparisons

| Group | Frozen state | Query | Small decoder |
|---|---|---|---|
| A | real | original | original Flat/task head, no fit |
| B_local | unused | unused | fit on Local + zero history + availability |
| B_donor | matched other conversation | recipient original | fit |
| B_real | real | original | fit |
| C_donor | same matched donor | corrected recipient | freeze exact selected B_donor |
| C_real | real | corrected recipient | freeze exact selected B_real |

B: LayerNorm → Linear64 → GELU → Linear7, identical dimensions (2307 inputs),
initialization, train sample order, steps and budget for all three groups. A
train-only feature normalizer is fixed in C. Base/Gap use only their effective
512 forward dimensions. Inactive Gap is hard-masked before and after normalization.

C: shared MLP128 over current Local, original Query, availability, type16 and
head16 embeddings; zero-initialized 64-delta; normalize original Query + delta.
Frozen pre-write Memory and current visible Keys feed the original differentiable
ridge/residualized read equations. No gradients to state, Keys, Local or source.
No extra write, no extra inference path in the source model.

Donors: same split, different conversation, exact current availability and
history-presence match, no label dependence. Recipient queries/current Keys read
the donor state. Common matched rows are used for ALL fits and paired metrics;
coverage and the separate full A reference are saved. Donor is an intentionally
invalid-history control, not a real observation trajectory.

## Fixed diagnostic protocol

30 epochs per stage, Adam1e-3/weight_decay1e-5, batch256, clip1, seed66. Original
task is 7-class CE. Earliest test-best ACC includes epoch0; associated UA/W-F1
are reported at that point, and last results are also saved. A remains unchanged.
No Optuna, no hyperparameter/threshold sweep. All optimization labels are TRAIN
labels; test labels select this internal diagnostic's checkpoints only.

Source: original GCNet-random-mask-v1, emotion-only causal Flat1600, Local Skip
retained, seed66 per-rate test-best checkpoints. Train features use one fixed
rate-specific offline GCNet mask view (mask seed66+rate_index) sampled with the
exact original host/guest padded-rectangle algorithm and speaker selector.
This offline probe view is NOT the source's cyclic mask training. Test features
use winning-epoch saved masks from `exact/source_0`, never final-retest masks.

All source weights are protected. Full states exist only in RAM/GPU; outputs
preserve source/mask/data hashes, configuration/commit, masks/IDs/donors, all
selected predictions, best/last diagnostic weights/optimizer/RNG. No original
checkpoint or historical experiment directory is overwritten.

## Commands (biggpu)

Python: `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Run each rate with its own directory, on a healthy UUID (host GPU4 forbidden):

```bash
CUDA_VISIBLE_DEVICES=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
  /data2/yb/conda_envs/cmerc-repro/bin/python \
  experiments/osram_meld_query_decision_20261014/run.py \
  --source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline \
  --selected-masks /data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/exact/source_0 \
  --dataset-root /data2/yb/paper/remote_runs_20261012/data \
  --meld-pickle /data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 --rate 0.7 \
  --output /data2/yb/remote_experiments/osram_meld_query_decision_20261014/rate_0p7
```

For a partial true-MELD smoke append `--max-batches 1 --epochs 2`, use a separate
output. It is never aggregated as a completed score. Four fits can run concurrently
on a healthy V10032GB only after peak memory/resource admission; no unrelated
jobs are stopped. Each rate sequentially fits its dependent B→C stages.

```bash
python experiments/osram_meld_query_decision_20261014/analyze.py \
  --runs /data2/yb/remote_experiments/osram_meld_query_decision_20261014 \
  --output /data2/yb/remote_experiments/osram_meld_query_decision_20261014/analysis
```

Interpretation: B_real beating Local/Donor gives evidence of usable readout signal
for this fixed head family. C improving its frozen B while outperforming donor
capacity controls supports a correctable Query limitation. Failure of either
restricted learner is not a proof of global optimality/no information. D simple
raw-history controls and multi-seed experiments are not launched in this round.
