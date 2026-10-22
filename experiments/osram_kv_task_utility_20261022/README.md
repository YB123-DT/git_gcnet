# Joint-write KV fit versus future task utility

INTERNAL DIAGNOSTIC ONLY. Frozen MOSI seed66 original no-JEPA LargeFlat.
No training, no selection of a new model, no change to OSRAM or missing masks.

## Predeclared scope

- One protected original cyclic Flat checkpoint, its previously selected miss0.3
  BEST; hold these weights fixed for all eight rates0.0–0.7.
- Original `gcnet-random-mask-v1` exact test batch-mask banks, metadata/label
  alignment checked; no new random masks and no persistent augmentation.
- Every valid joint source write with a valid target1/2/4/8 utterances later.
  No pair selection using labels. No separate per-head/per-modality task rows.
- All real labels enter MSE; exclude neutral labels only for binary polarity
  correction/harm counts. These pair-slot counts are not full-test W-F1 gains.
- biggpu, unchanged MOSI `s0` Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2.
  GPU4 forbidden. One GPU process on an idle healthy card; no existing jobs stopped.

## Measurements

At the actual source write, capture its already-decayed input `D=alpha*Mprev`,
actual availability-masked keys/values, and actual post-write memory. Apply
the original learned `sqrt(beta)` slot scaling to both K and V:

```
E_before = ||Vbar - D @ Kbar||²_F / max(||Vbar||²_F, 1e-12)
E_after  = ||Vbar - Mpost @ Kbar||²_F / max(||Vbar||²_F, 1e-12)
Q        = E_before - E_after
U[t,j]   = (y[j] - pred_omit[j])² - (y[j] - pred_keep[j])²
```

E and Q pool all heads and active columns for each joint write. Writes with
energy<=1e-12 are counted and excluded from association analysis. Q is not a
relative percent reduction. U>0 means retaining that joint write helped the
specified future target, not that it recovered a missing modality.

The factual fork starts at Mpost; Omit starts at D. Thus only that source
write is omitted; its decay and current read are preserved. At every later
valid timestep, both forks decay, read before write, and execute the original
joint Block Delta Write using the same real K/V. Padding never decays/writes.
Captured Local, queries and current keys are reused; the original Flat, Skip,
LayerNorm and scalar task head are unchanged. This is valid for this original
model because these projections do not depend on its evolving memory state.

The old retention audit's `decay_only` branch omits subsequent writes; it is
NOT the Omit intervention defined here. Nor do we bypass the production
write-mask guard to introduce an all-missing current utterance.

## Checks and outputs

Unit tests compare forks to a complete causal scan that truly skips the selected
write, including source prediction invariance, padding, weighted KV formula,
zero-write-step negative control, and no RNG/parameter change. A real GPU run
also compares factual predictions against an unwrapped normal forward, its
saved miss0.3 predictions, and a direct full-model omit scan for source0.

`ROWS.jsonl.gz` stays remote (per-pair scalar records, no features/weights).
`STATUS.json` records source, checkpoint/data/mask hashes, configuration,
environment, GPU UUID, parity maxima, elapsed time, peak CUDA allocation.
`REPORT.json`, `RESULT.md`, `AGGREGATE.csv` summarize all rates, distances and
current patterns, Q quartiles and median split, and conversation-clustered
descriptive bootstrap intervals. Repeated pair slots are not independent.

Example (bind the recorded healthy physical GPU UUID):

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_kv_task_utility_20261022.run \
  --reference /data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003 \
  --gpu-uuid GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
  --source-commit SOURCE_SHA --output /path/to/fresh/output
```

Add `--smoke` for one real test batch at0.3. Full scope is the eight-rate test
set, not another 150-epoch training. Run from the immutable recorded source.

## Interpretation limits

This checkpoint was historically chosen using Test-oracle; the audit is not
independent generalization evidence. Internal association fit need not imply
emotion relevance. Correlation can reflect value magnitude, update magnitude,
history position, or missing patterns; grouped results are descriptive, not
causal mediation. Strong KV improvement alongside negative/zero U is evidence
against KV fit as a sufficient task-value proxy, not proof the write equation
is defective or a new writer will improve performance. Do not add modules or
expand seeds based only on this audit.
