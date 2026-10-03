# Evidence-centered hierarchical decision correction

INTERNAL DIAGNOSTIC ONLY

Status: implementation/verification in progress; no training result yet.

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
