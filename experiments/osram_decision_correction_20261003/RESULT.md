# Evidence-centered hierarchical decision correction

INTERNAL DIAGNOSTIC ONLY

Status: implementation verified and all three seeds launched on biggpu GPU0;
training in progress, no final performance result yet. Code commit: 2b00639.

## Fixed experiment

MOSI seeds 66/67/68, 100 epochs each. Reference configs/checkpoints:
`/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_{66,67,68}`.
Current pre-change implementation: 29a9d7c. Original random cyclic rates 0.0–0.7,
batch32, Adam lr0.001, weight decay0.00001, original masks, MSE task primitive and
per-rate BEST Test-oracle selection stay unchanged. The new option
`--osram-decision-correction` replaces the Flat decision path and changes task
supervision to the fixed mean of three exits. It does not enable the previous
Gap-increment Filter, Relation, gate, completion, JEPA or paired-view training.

## Architecture

One unchanged causal Memory scan supplies Local and Base/Gap. Use causal forward
512-dimensional contexts, not their zero backward halves. Three separate MLPs
use LayerNorm(input) -> Linear(input,128) -> GELU -> Linear(128,C), no dropout.
H_L is normally initialized; D_B/D_G output weights zero-initialized and output
bias absent because a shared final bias would cancel in the difference.

```
sL   = H_L(L)
dB   = D_B(L,B) - D_B(L,0)
sLB  = sL + dB
dG   = D_G(L,B,masked_G,a) - D_G(L,B,0,a)
sLBG = sLB + dG
task = (original_task(sL,y) + original_task(sLB,y) + original_task(sLBG,y))/3
```

Both evaluations of a correction share weights and retain gradients. Padding,
inactive Gap and empty history are hard masked. Active parameters train jointly
in one stage. Old Flat/head must not execute in this path. Inference uses sLBG;
there is no label-dependent routing. Initialization recovers the NEW Local-only
head, not original Flat. Centering is not a guarantee of useful corrections.

## Execution

Server: biggpu. GPU4 prohibited. Prefer GPU0 for same-card parallel seeds,
resource permitting. All output directories/logs/snapshots are independent of
the completed Gap Filter experiment. Launch only after correctness tests and
specification/quality reviews pass. Final source/config/environment/GPU/reference
hashes go in PROVENANCE.json and measured active/retained counts in PARAMETERS.json.

Runner command from the isolated snapshot:

```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u experiments/osram_decision_correction_20261003/run.py --output /data2/yb/remote_experiments/osram_decision_correction_20261003/seed_66 --seed 66 --gpu 0 --commit CODE_COMMIT
```

Repeat for 67/68 with their own output paths. This is from-scratch training,
not checkpoint resume. Eight BEST checkpoints and original mask hashes are
required per seed; 100-epoch history and successful provenance are required
before reporting completion. Main scores use Full output's per-rate BEST.
Local/LB exit metrics at the same Full-selected checkpoint are diagnostics,
not independently selected oracle outputs.

## Baseline summary

| Seed | Original Flat 8-rate W-F1 (%) | High (.5/.6/.7) |
|---|---:|---:|
| 66 | 81.068 | 76.352 |
| 67 | 80.556 | 75.990 |
| 68 | 80.053 | 74.440 |
| Mean | 80.559 | 75.594 |

New model results pending. Do not interpret tests or successful launch as a
performance improvement. No additional configuration sweep is authorized.

## Measured capacity and pre-launch checks

| Model | Stored total parameters | Trainable | Retained inactive |
|---|---:|---:|---:|
| Original Flat | 13,509,793 | 13,509,793 | 0 |
| Decision correction | 13,943,592 | 3,992,487 | 9,951,105 |

The new decision head contains 433,799 parameters. Old Flat/readout parameters
remain in the state dict for compatibility but are never executed and are
excluded from the optimizer. This is NOT a parameter-matched comparison; the
active decision architecture is substantially smaller than original Flat.

Default-off tests compare model AND OSRAM directly with pre-change source
29a9d7c, including RNG, common parameter initialization and nonzero Flat outputs.
Standalone tests verify learned independent B=0/G=0 cancellation in train/eval,
C=1/C=6, fixed conditioners, differentiable paired calls, safe masked NaNs,
single scan, unchanged Memory tensors and strict checkpoint reload. Training
tests verify equal task losses and original evaluation thresholds/valid labels.

Actual cfg84-dimension synthetic CUDA smoke on biggpu V100: two optimizer steps,
one subsequent evaluation, finite parameters/gradients and exact first-valid
Local=LB=Full. Synthetic batch=4, sequence length=10; peak allocated CUDA memory
164.2 MiB. This is a correctness smoke, not a full-batch memory capacity claim
or a model performance result. See cuda_smoke.json and verification logs.

Fresh verification: 29 local tests and 33 remote V100 tests passed; scoped
compileall and git diff --check passed. Remote archive excludes three Git-object
legacy tests that passed locally. Specification and code-quality reviews passed
without blocking findings. Six core/runner source SHA256 hashes matched local
files before launch. GPU0 was checked idle with 32,495 MiB free.

## Launch record

Remote root: `/data2/yb/remote_experiments/osram_decision_correction_20261003`.
Immutable source snapshot: `code/`. GPU0 UUID:
`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45` (process-visible cuda:0).

| Seed | PID | Log | Output |
|---|---:|---|---|
| 66 | 3995805 | seed_66.log | seed_66/ |
| 67 | 4050144 | seed_67.log | seed_67/ |
| 68 | 4104634 | seed_68.log | seed_68/ |

Started 2026-10-03 at 13:26 UTC, staggered by first-epoch verification and fresh
GPU identity/free-memory checks. Independent persistent processes run concurrently;
this is not serial training. Logs/checkpoints are separate. See launch.json for
exact commands/timestamps. Earlier SSH transport resets occurred before formal
launch; no training process was duplicated or migrated to a local GPU.
