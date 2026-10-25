# 12-slot Source–Role MoE: two MOSI connections

INTERNAL TEST-ORACLE SCREEN. This is an exploratory two-arm experiment, not
MARS reproduction or an independent test-set generalization claim.

## Fixed design / approved scope

Only MOSI seed66, from scratch, cmerc-repro on biggpu. GCNet random-mask-v1,
cyclic rates 0–0.7, batch32, Adam, WD1e-5, MSE, original splits/dropout;
150 epochs, LR3e-4 through epoch100 and 3e-5 thereafter. No auxiliary loss,
new queries/writes, teacher, Nested, additional seed or parameter search.

Keep 12 distinct slots (Base/Gap-A/T/V × historical Value source A/T/V).
Each slot has a shared 512→64 projection and role/source identities; unavailable
roles/sources and padding are zeroed. Concatenation preserves slot identities:
no source summation or Gap averaging before the experts.

All four independent experts receive the same full slot vector, Local64,
availability3, strict-past trajectory12, and explicit active-slot mask12.
Each expert is an MLP with hidden256 and output128. A shared router with
hidden128 produces four softmax weights. Their weighted nonlinear outputs
feed a shared zero-initialized 128→1600 projection. No predefined expert roles.

* `residual`: original Local Skip + original Flat Adapter + expert output → LN → head.
* `replace`: original Local Skip + expert output → LN → original head.

The unused original Adapter in replace is retained in the state dictionary but
frozen and bypassed (its allocation preserves original initialization order).
Report allocated and trainable parameters separately. The two arms have identical
MoE initialization; they intentionally differ in active Flat capacity.
Replacing the Adapter also bypasses its Dropout: the two arms share initialization,
data order and availability masks, but not an identical sequence of training-time
CUDA Dropout draws. Do not call this a fully stochastic-operation-matched ablation.
With no history, source slots are zero; experts can still process current Local.
With ATV visible, Base source slots remain active. These are not V3's missing-only gates.

## Execution plan and checks

1. Test slot masks, distinct inputs, expert independence, gradients, zero output,
   disabled legacy behavior, initialization/RNG, one causal scan and reload.
2. Reuse accepted Flat's exact test masks and metadata, data hashes and original
   initialization. Real GPU smoke: multiple MSE updates, upstream expert/router
   gradients after the zero output opens, finite loss and strict reload equality.
3. Freeze a source snapshot and launch only the two arms, separately logged.
4. Every epoch verify training mask digest/update count/LR against the original
   reference. Retain eight same-call test-best checkpoints/predictions/masks and
   full last Adam/RNG/epoch recovery. Select strictly by per-rate test W-F1.
5. Report eight-rate and high-missing ACC/W-F1/MSE, per-rate epochs, resource cost,
   router usage/entropy and expert-output similarity (not evidence of mechanisms).

Reference: `/data2/yb/remote_experiments/osram_mosi_cmerc_20261023/execution_v2`:
Flat seed66 eight-rate ACC79.535061 / W-F179.460447; high W-F173.351571.
This uses the new Missing protocol, not the old 80.7 reference.

## Commands

Use the immutable source snapshot on biggpu; GPU4 is prohibited.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -m experiments.osram_source_moe_mosi_20261025.run --mode residual --phase smoke --gpu-uuid HEALTHY_UUID --source-commit SHA --output UNIQUE_SMOKE
/data2/yb/conda_envs/cmerc-repro/bin/python -m experiments.osram_source_moe_mosi_20261025.run --mode replace --phase smoke --gpu-uuid HEALTHY_UUID --source-commit SHA --output UNIQUE_SMOKE
/data2/yb/conda_envs/cmerc-repro/bin/python -m experiments.osram_source_moe_mosi_20261025.run --mode residual --phase train --gpu-uuid HEALTHY_UUID --source-commit SHA --output UNIQUE_RUN
/data2/yb/conda_envs/cmerc-repro/bin/python -m experiments.osram_source_moe_mosi_20261025.run --mode replace --phase train --gpu-uuid HEALTHY_UUID --source-commit SHA --output UNIQUE_RUN
```

Resume only with `--resume` and exactly matching configuration/source identity.
Source-derived tensors are Value-slot contributions, not pure modality causality.
