# Source-role tokens: controlled MOSI / MOSEI redo

INTERNAL TEST-ORACLE SCREEN. Finite user-authorized seed66 comparison.

## What changes, and what does not

The previous `source-mean` experiment followed the original requested
`MaskedMean` over active Gap roles. That mean was not an implementation bug.
This redo tests a different representation: keep the 12 role/source slots
independent until Attention, rather than averaging Gap-A/Gap-T/Gap-V first.

Token order is role-major: Base, Gap-A, Gap-T, Gap-V, each containing historical
Value-source A/T/V. Each slot becomes a 128-dimensional token with explicit role
and source identity. The shared token MLP receives87 dimensions: role-projected
read64, role embedding8, source embedding8, availability3 and trajectory4.
It is `LayerNorm87 → Linear128 → GELU → Linear128`; count/age are log1p encoded.
The four512→64 role projections share their parameters across historical sources.
One source-role SelfAttention128/4heads is followed by current-Local
CrossAttention. Inactive Gap roles, never-observed sources and padding are
excluded. No-history rows are skipped safely.

The output still uses the same zero-initialized projection into the original
Flat pre-norm residual. This is an explicitly fixed controlled-comparison choice,
not a claim that the user originally specified the final residual connection.
It does not replace the entire readout. This is not a pure one-factor mean
ablation: the previous263-d pair-feature encoder is replaced by87-d per-slot
encoding, with role embeddings and a different parameter count. Any score
change cannot uniquely establish that the previous mean caused the loss. Original
Flat1600, Local Skip, task head and task MSE remain unchanged. Neither model has
an auxiliary loss, new missing schedule, second Memory trajectory, learned Gap
Query, completion, JEPA or source-scaling gamma.

The CHM Value-source decomposition is reused; its meaning is joint-write Value
slot contribution, not pure modality semantics or causal semantic attribution.

## Protocol

| Dataset | Seed | Epochs | LR | Batch | WD |
|---|---:|---:|---|---:|---:|
| MOSI | 66 | 150 | 3e-4 epochs1–100; 3e-5 epochs101–150 | 32 | 1e-5 |
| MOSEI | 66 | 60 | constant1e-3 | 32 | 1e-5 |

Both retain `gcnet-random-mask-v1`, original cyclic0.0–0.7 missing, same reference
data/split and exact evaluation masks, s0 environment and Adam/task loss. Each
rate uses its own earliest strict TEST-best W-F1 checkpoint. Means therefore
combine different epochs; they are not single-checkpoint deployment scores.
Eight BEST weights and matching predictions/masks plus complete last
model/optimizer/RNG recovery are retained in new independent output directories.

Reference Flat seed66 completed W-F1 mean/high: MOSI79.249087/73.409792;
MOSEI83.279076/80.181722. These are historical matched-protocol references, not
scores measured for the new module. The prior MOSI source-mean completed at
78.895455/72.612688; retain that experiment rather than overwriting its records.

## Reproduction entry point

Reuse the existing finite runner and training loop with an explicit layout:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_source_relation_attention_20261021.run \
 --layout role-tokens \
 --output /data2/yb/remote_experiments/osram_source_role_tokens_20261021/execution_v1 \
 --source-commit IMMUTABLE_SOURCE_SHA \
 --gpu-uuid HEALTHY_GPU_UUID
```

The controller first performs finite real-data MOSI/MOSEI GPU smoke, then runs
only these two jobs concurrently if actual GPU/RAM headroom permits. Never use
biggpu hostGPU4. Never interrupt peers or change batch size to force concurrency.

The default `--layout source-mean` remains the legacy implementation; the new
model is selected only with `--layout role-tokens`. All new layout source hashes,
including module, worker, controller and report, are included in run identity.

## Verification and reporting

The runner verifies original parameter initialization, exact zero-init selected
batch predictions, masks/config/data, every training epoch mask/update/LR trace,
one Memory scan per training batch, optimizer/RNG recovery, original and added
parameter updates, and named upstream gradient coverage. The role embedding
must have nonzero task gradient after the output projection opens. Each actual
active Base/source and Gap-target/source input path is checked separately;
unobserved/inactive paths must not be described as verified gradient paths.

CrossAttention diagnostics retain a four-by-three `role_source_attention_mean`
matrix, not a three-source vector formed by losing the role axis. Attention
weights and internal norms do not prove task contribution or emotion semantics.

The new module has530,470 parameters (old source-mean553,678); the unchanged
original network has13,509,793, giving14,040,263 total. Counts were measured by
`sum(p.numel() for p in module.parameters())` in s0; smoke will verify the full
model count. Measured smoke cost, immutable SHA, deployment identifiers
and actual results will be recorded after verification. The reporter includes
only completed training jobs, names this layout `SourceRoleTokens`, and publishes
per-rate ACC/W-F1/MSE and mean/high summaries automatically. No extra seeds,
hyperparameter search or new mechanisms are authorized by this finite redo.
