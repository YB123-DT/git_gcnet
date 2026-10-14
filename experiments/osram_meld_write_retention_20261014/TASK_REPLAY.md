# Original Flat fixed-target-Query four-state task replay

INTERNAL DIAGNOSTIC ONLY

This extension performs frozen inference only. It does not replace or retrain
Flat, generate masks, tune a threshold, change Query, or change the Memory write.
Source is the retained seed66 reproduction of historical commit `b4a3138`:

```
/data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline
```

Only its eight original large-Flat checkpoints and stored canonical
`gcnet-random-mask-v1` final-evaluation masks are accepted. No legacy/no-Skip
substitution and no small-Flat substitution. These masks are the existing
write-retention audit reference, not selection-epoch reconstructed masks. Task
cohort scores are not the reported selected-epoch whole-test performance.

## Data flow

Run the original model exactly once per batch with `WriteCapture` and
`TaskCapture`. The latter wraps the actual `_scan` call and captures the four
original task queries and its output reads; `FlatReadoutCapture` captures Local
at `local_skip` and the adapter input. `_scan` has no standalone read API, so
`original_read` transcribes only its read algebra, using its original
`_address_residual`, ridge and residual-strength attributes. No re-projection,
new normalization, new query or new current-state computation is introduced.

For source `s`, all four branches start from the same **real joint-write
post-state**. Existing `audit.replay_forks()` gives:

- Normal: original retention and subsequent joint writes.
- Decay-only: retention, but no subsequent writes.
- Write-only: subsequent original joint writes, but no retention.
- Hold: neither subsequent writes nor retention.

The target state is after any target retention but before target write. Fixed
target Local, four original queries and observed keys enter the exact original
Base/Gap read, forward512/backwardzero512 interface and unchanged Flat adapter,
Local Skip, emotion LayerNorm and `smax_fc` task head. Inactive Gap remains zero.

Distances **2/4/8** are headline analyses. Distance1 is sanity-only: Normal must
equal Decay-only and Write-only must equal Hold at the task-logit level. Normal
must reproduce original logits (numerical tolerance), exact predicted classes,
stored prediction classes and stored labels. State hashes/absence of gradients
and CUDA RNG are checked. The Torch CPU DataLoader may consume a base-seed draw;
this is logged separately, not claimed to be stochastic branch computation.

## Pairing and metrics

Eligibility uses observed source modality and valid source/target utterances,
never labels. Every branch has exactly the same source-target population. All
source modalities share the same joint-write fork: compute target logits once
per source/target/branch, then attach each source-modal association probe.

Rows contain source-head-mean relative recovery error, same-source immediate
post-write error, original trajectory degradation, paired association-error
change, candidate-centered cosine/retrieval where available, original/branch
task logits, CE, ΔCE, corrections and harms. Positive ΔCE means task harm.

`task_analyze` reports per rate/distance/current availability/source modality
and normal association degradation strata. It computes fixed-seven-class UA,
ACC, W-F1, class recall/support and direction tables/correlations on paired
cohorts. Main task tables deduplicate source modalities by conversation/target
at fixed rate and distance, so ATV sources do not receive threefold weight.
Across distances a target may recur: no pooled deployment score is invented.
Source-Text subsets remain available with their own explicit cohort sizes.

Compare association error change and target CE change on **the same pair**.
Do not infer task harm from poor source reconstruction alone. Agreement,
disagreement and negligible task changes are all legitimate results. These
state interventions isolate fixed-Query state sensitivity; they do not establish
optimal Query retrieval or optimal classifier use of a given readout.

## Execution

Run on `ssh biggpu`, healthy host GPU only (GPU4 forbidden), a new output path:

```bash
CUDA_VISIBLE_DEVICES=GPU-HEALTHY-UUID /data2/yb/conda_envs/cmerc-repro/bin/python \
  -m experiments.osram_meld_write_retention_20261014.task_replay \
  --gpu-uuid GPU-HEALTHY-UUID \
  --source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline \
  --dataset-root /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138/dataset \
  --meld-pickle /data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --output /data2/yb/remote_experiments/osram_meld_fixed_query_task_20261014/full
```

Smoke uses `--rates 0.7 --max-batches 1` and an independent output. Analyze:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python \
  -m experiments.osram_meld_write_retention_20261014.task_analyze \
  --input /data2/yb/remote_experiments/osram_meld_fixed_query_task_20261014/full \
  --output /data2/yb/remote_experiments/osram_meld_fixed_query_task_20261014/analysis
```

Artifacts: compressed raw rows per rate, frozen-invariance/provenance SUMMARY,
paired CSV/JSON with class recall and source-Text direction statistics, and
cohort RESULT.md. Existing storage audit results are never overwritten.
