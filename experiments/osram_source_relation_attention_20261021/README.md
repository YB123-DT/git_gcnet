# Source Relation Attention: finite MOSI / MOSEI screen

INTERNAL TEST-ORACLE SCREEN. User-approved two single-seed one-stage runs.

Original CHM, observed encoder, Local/Skip, K/V/Query, joint Block Write and
Flat1600/task head stay unchanged. Reuse exact differentiable Value-source
tracking, not old Source-Role gamma/TASR. The module outputs a feature residual,
not scaling coefficients or alternative sentiment predictions.

For each source s and role r: `u_sr=P_r(read_rs)` with four bias-free512→64
projections shared across A/T/V. Base b; active-missing-target Gap mean g.
`[b,g,b*g,abs(b-g),availability,trajectory_s]`263→128→128 shared nonlinear MLP.
Trajectory logs count/age; coverage and ever-seen retain their definitions.
Three source-identity embeddings preserve A/T/V labels. One128-d/4-head
SelfAttention, then one128-d/4-head CrossAttention queried by projected Local.
The128→1600 output is zero-initialized and added to original Flat pre-norm.
No extra Transformer FFN, internal residual/norm, new loss or dropout. Original
Flat dropout is unchanged. Init is isolated with fork_rng. No-history/padding
are zero after output bias; never-seen sources and inactive Gaps are safely
masked before transforms. All-masked attention rows are skipped.

New opt-in switch `--osram-source-relation`, defaultFalse. Cannot combine old
source-role/TASR or non-plain causal Flat options. No default numerical/RNG path
changes. Original checkpoint loading remains strict with switch disabled.

| Dataset | Seed | Epochs | LR | Batch | WD | Reference seed66 W-F1 mean/high |
|---|---:|---:|---|---:|---:|---|
| MOSI |66|150|3e-4 first100;3e-5 final50|32|1e-5|79.249087 /73.409792|
| MOSEI |66|60|constant1e-3|32|1e-5|83.279076 /80.181722|

Both use original MSE/Adam/cyclic0.0..0.7 and gcnet-random-mask-v1. Test WF1
strict-best/earliest tie per rate, same-call ACC/predictions/mask,8BEST+full
last optimizer/RNG. No validation holdout, new seed, lr search, Gate, Nested,
JEPA, completion, persistent views or InfoNCE. Do not mix historical MOSI80.7
different-mask results or MOSEI1e-4 runs into matched comparisons.

Runtime: biggpu original s0, not cmerc. HealthyGPU7 UUID, never hostGPU4.
Controller first runs two finite3-batch maximum real-data GPU smoke jobs,
checking named upstream module+source-input gradients, zero-init predictions,
original updates, full recovery, then starts two independent trainings
concurrently on the same card if measured headroom permits. Immutable source,
independent outputs/logs, no peer kills. Final report runs automatically.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_source_relation_attention_20261021.run \
 --output /data2/yb/remote_experiments/osram_source_relation_attention_20261021/execution_v1 \
 --source-commit SOURCE_SHA \
 --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

For full-state recovery use the saved worker command plus `--resume`; do not
start a second controller over an active output. Do not call loading-only BEST
weights a full resume. Reporter only ranks complete jobs; failure remains shown.

```
python -m experiments.osram_source_relation_attention_20261021.report --output OUTPUT
```

Parameter addition553,678 (same architecture across both datasets). Pure CPU
tests cover module, exact old disabled outputs/backward/RNG, zero-init and real
source graph connection. Remote snapshots omit .git, so two older git-show-based
historical tests require the actual Git worktree and are excluded there; the new
legacy regression explicitly loads the frozen pre-change osram.py instead.
No claims that source relations are pure modality semantics, reliability or
emotion shift. Successful smoke establishes implementation, not task gains.
