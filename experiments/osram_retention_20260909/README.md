# Real checkpoint retention evaluation

No training, no parameter changes. Source instrumentation commit 7d7b239.
All inputs use existing frozen features, official test loader and the same
`_schedules(config, 'test')` / `_prepare_view(..., epoch=0)` evaluation path.
The model is strict-loaded from the actual forward-only (zero-slot) checkpoint.
Diagnostic collection uses a temporary OSRAM pre-hook, removed after each call.

Order executed: IEMOCAP4 seed66 rate0.7 first → check JSONL and continuity →
seed66 rates0.1/0.3/0.5 → check paired direction → seeds67–70 four rates →
MOSI seeds66–70 four rates. No overlap-root-cause interpretation.

Each checkpoint is the existing eight-rate-mean Test-oracle checkpoint, not a
newly selected checkpoint or a per-rate checkpoint. This diagnostic is not an
F1 benchmark or formal paper result. Checkpoint hash/epoch/selection policy,
mask hash, record count and continuity checks are stored per run in metadata.json.

The first real JSONL is
`raw/iemocap4_seed66_rate0.7/retention.jsonl.gz` (lossless gzip in git).
Local/remote uncompressed retention.jsonl is also retained.
Decompress with `gzip -dc retention.jsonl.gz > retention.jsonl` for external analysis.

Continuity check compares post(t) to pre(t+1) for the SAME last probe during
consecutive missing steps, per head: error, cosine and norm_ratio. This is a
metric-level continuity assertion, not a claim that arbitrary vectors are
identified by these three scalars. All currently collected assertions pass at
zero difference. First-batch classification logits with/without instrumentation
are also checked for exact equality for each evaluation task.

Paired difference = write_damage - decay_damage for each retained query/head.
RESULT.md and paired_difference.csv contain means, medians, positive fractions
and conversation-balanced summaries; heads/time points are not independent
samples. These are local single-step changes relative to the last REAL observed
value, not cumulative attribution over the whole missing interval. Thus even
uniformly positive paired differences do not prove write causes the F1 gap,
nor that decay is harmless or unnecessary. No p-values are inferred from heads.

Example (run from repository root on biggpu, existing official Python):

```bash
PYTHONPATH=. /data2/yb/reproduction_envs/s0/bin/python3.10 \
  -m gcnet_missing_m3.evaluate_memory_retention \
  --checkpoint /data2/yb/remote_experiments/osram_forward_only_iemocap_20260908/iemocap4/seed_66/best.pt \
  --feature-root /data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP/features \
  --rate 0.7 --output-dir /tmp/retention_new_output
```

The output directory must not exist, preventing accidental overwrite.
Rebuild paired report: `python experiments/osram_retention_20260909/summarize_paired.py`.
