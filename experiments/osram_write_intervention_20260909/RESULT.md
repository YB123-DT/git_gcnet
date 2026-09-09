# Frozen-checkpoint causal write intervention — first pilot

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

IEMOCAP-4, Session/fold 5, seed 66, existing forward-only zero-slot
checkpoint epoch 33. No training, checkpoint reselection, or tuning.
The historical checkpoint used eight-rate-mean Test-oracle selection.
This pilot evaluates five rates, not the full eight-rate benchmark.

## Task result

Weighted-F1 (%); all three modes use identical masks and checkpoint weights.

| Missing rate | Reference | Protected write | Norm-matched global |
|---|---:|---:|---:|
| 0.0 | 84.5439 | 84.5439 | 84.5439 |
| 0.1 | 84.0821 | 84.1692 | 84.0821 |
| 0.3 | 81.5803 | 82.1713 | 81.8203 |
| 0.5 | 80.3239 | 80.3944 | 80.3951 |
| 0.7 | 78.3912 | 78.2353 | 78.7239 |
| Five evaluated rates mean | 81.7843 | 81.9028 | 81.9131 |

Protected minus reference: +0.1185 percentage points across these five rates.
Protected minus global: -0.0102 points. At rate 0.7, protected minus reference
is -0.1559 points and protected minus global is -0.4886 points.
These are a single seed's diagnostics, not significance or robustness claims.

## Association retention

Means below pool valid historical probe/head records, not independent samples.
No-history queries are excluded from errors. Signed damages are retained.

| Rate | Mode | Error at actual read | Error after write | Write damage |
|---|---|---:|---:|---:|
| 0.1 | reference | 0.055371 | 0.341871 | 0.286500 |
| 0.1 | protected | 0.048048 | 0.048112 | 0.000064 |
| 0.1 | global | 0.062403 | 0.314598 | 0.252195 |
| 0.3 | reference | 0.103056 | 0.329898 | 0.226842 |
| 0.3 | protected | 0.075832 | 0.075876 | 0.000044 |
| 0.3 | global | 0.113847 | 0.305764 | 0.191917 |
| 0.5 | reference | 0.156302 | 0.333961 | 0.177659 |
| 0.5 | protected | 0.108714 | 0.108745 | 0.000031 |
| 0.5 | global | 0.168785 | 0.312527 | 0.143742 |
| 0.7 | reference | 0.214040 | 0.359064 | 0.145024 |
| 0.7 | protected | 0.148237 | 0.148262 | 0.000025 |
| 0.7 | global | 0.229050 | 0.340376 | 0.111326 |

Protection strongly reduces measured write distortion, as its construction
intends. It also reduces error at subsequent actual reads. Neither establishes
that the retained historical content improves emotion recognition. At high
missing rate the task result moves in the opposite direction in this seed.
Possible explanations (not diagnosed here) include obsolete associations,
restricted useful new writes, and inference-time distribution shift.

Do not claim direction-specific task benefit from this pilot: protected does
not outperform the global control on the aggregate. Do not infer that a trained
protected-memory model must fail either. No new training or automatic tuning
was initiated after these results.

## Verification and provenance

- `iemocap4_seed66/metadata.json`: full config, checkpoint SHA256, epoch,
  per-rate/mode metrics, mask hashes, retention and write summaries.
- Checkpoint SHA256:
  `79aef3dab7d61aa41dabe5bd9f5cf8044f8c5fbf9c556684986e485f5a7be461`.
- 15 evaluations completed; CPU, no optimizer or GPU training.
- Same rate's mask hashes identical across all three modes. Hashes include
  padded batch availability, matching the retention evaluator; do not compare
  directly with training's valid-only mask hashes.
- Complete-input logits exactly equal in all three modes, all evaluated batches.
- All state-dict tensors unchanged after evaluation; no new parameters/buffers.
- Maximum same-state protected/global update norm mismatch: 3.8146973e-6
  (FP32; epsilon and rounding). Separate rollout trajectories are not guaranteed
  to have identical update norms at later steps.
- Existing/new relevant tests: 200 passed, one existing PyG deprecation warning.
- Syntax compilation and `git diff --check` passed.
- Raw per-head retention and per-write audit records are gzip JSONL files under
  `iemocap4_seed66/`. `err_post` describes post-write memory and can only affect
  future reads; it is not current-utterance classification error.

## Reproduce (evaluation only)

From the remote repository with its existing official Python environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
/data2/yb/reproduction_envs/s0/bin/python3.10 \
  -m gcnet_missing_m3.evaluate_write_intervention \
  --checkpoint /data2/yb/remote_experiments/osram_forward_only_iemocap_20260908/iemocap4/seed_66/best.pt \
  --feature-root /data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP/features \
  --output-dir /data2/yb/remote_experiments/osram_write_intervention_20260909/iemocap4_seed66_repeat \
  --device cpu
```

The output directory must not already exist. The default five rates are
0.0/0.1/0.3/0.5/0.7. The runner streams diagnostics without retaining all
historical keys and never modifies the saved checkpoint.
