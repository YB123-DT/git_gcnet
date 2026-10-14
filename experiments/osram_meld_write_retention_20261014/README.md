# Frozen MELD write and retention audit

INTERNAL DIAGNOSTIC ONLY. No optimizer or training. Source: original MELD
cfg84 causal no-JEPA Flat seed66, `gcnet-random-mask-v1`, eight per-rate
Test-oracle checkpoints. Reuse the saved selected-speaker masks exactly;
prediction and label parity must pass before interpreting any audit.

The two authorized diagnostics are:

1. Record each real source Key/Value before and after its joint Block Write;
   retrieve with the same source Key at strictly future pre-write distances
   1/2/4/8. Only within-conversation observed sources and valid targets enter.
2. From the identical source post-write state, replay normal, decay-only,
   write-only (retention=1), and held states. Use original jointly written
   A/T/V columns and the original `block_write`; no sequential modality writes.
   Every normal fork must reproduce the captured original pre-write state.

No backbone file is modified. Capture wraps the actual `block_write` call and
is restored before replay; audit uses detached tensors under `no_grad`.
Current output parity and model state hashes are checked. The first future
read occurs before the next write: at distance1, normal=decay-only and
write-only=hold. This is required by read-before-write, not an audit failure.

Metrics: norm-normalized reconstruction error, norm ratio, raw cosine;
centered cosine and candidate retrieval on a fixed four-value source pool.
The pool contains the source and its three latest strictly earlier observed
values of the same modality in the same conversation. Its mean is the
centering reference; this is **candidate-centered**, not train-mean-centered
cosine. Pools never change as source distance increases. Sources with fewer
than four candidates have unavailable centered/retrieval metrics, with their
coverage counted. Similar candidate Values can cause ties; retrieval top1
uses fractional ties, MRR averages possible tied ranks. Wrong-Key control
uses the most recent distractor Key and still targets the original Value.

Distances have different available source populations due to conversation
ends. Report paired error change relative to that same source's immediate
post-write reconstruction, as well as absolute curves. No error threshold
defines "broken memory". Reconstructing stored Value is not evidence of
future emotion usefulness; forgetting may remove task-irrelevant information.

Literature motivation: [Gated DeltaNet §3.2](https://arxiv.org/html/2412.06464v3)
separates retention and interference with retrieval contexts;
[DeltaNet §4.1](https://arxiv.org/html/2406.06484v2) evaluates associative
recall. This frozen real-feature replay is our adaptation, not their experiment.

Run on biggpu, healthy GPU0 UUID (host GPU4 forbidden):

```bash
CUDA_VISIBLE_DEVICES=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
/data2/yb/conda_envs/cmerc-repro/bin/python -m \
experiments.osram_meld_write_retention_20261014.run \
  --source /data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/meld/meld_flat_s66 \
  --dataset-root /data2/yb/paper/remote_runs_20261012/osram-ecerc-meld/dataset \
  --meld-pickle /data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --output /data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66 \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
python -m experiments.osram_meld_write_retention_20261014.analyze --input /path/to/output
# Render an already completed CSV in an environment that has matplotlib:
python -m experiments.osram_meld_write_retention_20261014.analyze \
  --csv /path/to/AGGREGATE.csv --output /path/to/figures
```

Optional `--max-batches 1 --rates 0.7` marks output partial, never complete.
Missing exact source weights fail before GPU use or output creation. Restore
the original weights before running; alternate/no-skip/old-mask models are
not replacements for this source.

## Authorized baseline reproduction when weights cannot be recovered

The user subsequently authorized searching for backups and, if absent,
retraining the original baseline to enable the two audits. `reproduce.py`
automates historical-code training -> historical-score comparison -> both
frozen audits -> CSV/plots. The training snapshot must match original commit
`b4a313841f098487a6482d2712bdac94aa8e319e`; the exact original config is read
from the historical run. It retains all eight best weights. This is a new
reproduction, not a restoration of deleted original parameters.

Use a dedicated healthy GPU and immutable Git worktrees:

```bash
CUDA_VISIBLE_DEVICES=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
experiments/osram_meld_write_retention_20261014/reproduce.py \
  --training-code /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138 \
  --audit-code /path/to/current/audit/worktree \
  --source-config /data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/meld/meld_flat_s66/config.json \
  --historical-source /data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/meld/meld_flat_s66 \
  --dataset-root /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138/dataset \
  --meld-pickle /data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --output-root /data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138 \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
```

`PIPELINE.json` records each child PID/status/log/return code. Stage failures
stop dependent stages. `REPRODUCTION_COMPARISON.json` separates per-rate
selection scores from final saved-mask scores and checks mask/prediction
agreement with history. Actual audit always uses the reproduced model's own
saved masks and requires exact prediction parity with that model.
