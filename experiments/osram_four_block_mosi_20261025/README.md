# Four frozen-parent MOSI branches

INTERNAL TEST-ORACLE SCREEN

User specifications are the implementation source. The named reference package
was not present in this checkout or supplied as an attachment. This new package
implements the posted equations rather than claiming to have run an unavailable
reference archive. No old model or experiment is modified.

## Fixed parent and budget

- MOSI seed66, accepted cmerc-repro Flat, original GCNet random-mask-v1.
- One protected miss0.3 checkpoint (epoch75) shared across four experiments and
  all rates, frozen and in eval mode. Parent hash is checked and recorded.
- No retraining of parent. The new branches alone use Adam, batch32, WD1e-5,
  150 task epochs: first100 LR3e-4, last50 LR3e-5, existing gradient clipping.
- Original cyclic per-batch training rates and sampler retained. Branch RNGs
  are isolated from mask sampling. Test uses accepted exact mask banks.
- CSDI first fits rank64 PCA on TRAIN and performs150 denoising epochs with
  the same LR schedule; then freezes the denoiser and trains bridge150epochs.
  Final denoiser is used, without test-based denoiser selection.
- These finite defaults extend the existing MOSI budget; not optimized settings.
- Test-best per-rate W-F1 with same-call ACC/MSE/predictions/mask saved; all
  valid labels including zero participate in MSE. Metrics exclude gold-zero
  for binary ACC/W-F1 exactly as the existing trainer does.

The fixed-parent baseline is evaluated on the same eight masks. It must not be
confused with the previously reported eight-rate Flat Test-best average, which
uses different parent checkpoints at different rates.

## Interfaces

`osram_four_block_wiring/frozen_flat.py`: one original forward and read-only
emotion_norm pre-hook expose the1600-d pre-LN anchor, original hidden/logits,
Local256, Base/Gap1024, and visible modality latents. Internal context2304 keeps
all forward512 Base/Gap slots. Parent Norm/head weights are frozen but input
gradients are enabled when decoding a new delta. No second Memory scan.

`blocks.py`: NeuMiss80-d conditional iteration; strict-past8-utterance degree2
signature; TRAIN-neighbor FP64 ridge residual solver. NeuMiss context coordinates
are observed and no inactive Gap is treated as a missing raw coordinate.

Ridge bank contains frozen TRAIN hidden/labels/base predictions under legal
random masks for eight rates. Every query excludes its entire conversation and
requires same rate/pattern/parent. No query label enters retrieval. Selection is
hard topK, with gradients through selected embeddings and solve. Initialization
need not equal Flat; ridge raw parameter starts0 => softplus(0)+1e-4.

`csdi_latent.py`: conditional latent DDPM,20steps, channels64,4residual layers,
linear beta1e-4..0.02; explicit modality/step/context conditioning. This is a
self-contained latent adaptation, not a reproduction of the unavailable
reference denoiser architecture or its reported parameter count. TRAIN complete
features supply frozen-projector denoising targets, never enter CHM. Four
hypotheses remain separate through nonlinear bridge/Norm/head; scalar predictions
are averaged. Per-query keyed evaluation samples are cached once and reused.

## Execution and recovery

```
/data2/yb/conda_envs/cmerc-repro/bin/python -m experiments.osram_four_block_mosi_20261025.run --mode neumiss --output UNIQUE_PATH --gpu-uuid HEALTHY_UUID --source-commit SHA
```

Modes: `neumiss`, `signature`, `ridge`, `csdi`; independent processes, not a
串联. `--smoke` performs bounded real training/evaluation, not a performance
experiment. `--resume` reloads module/optimizer/epoch/RNG and denoiser stage if
needed. Parent, config and method identity must match. No existing checkpoint
is overwritten. Source is snapshotted before formal launches.

Files: SUMMARY/RESULT/history, eight branch BEST and matched predictions,
complete last state; Ridge support bank; CSDI PCA manifest, denoiser optimizer
state and fixed test hypotheses. Large artifacts remain on biggpu.

Formal scope: four methods, one seed, MOSI only, no HPO or automatic additional
seeds. Three zero-exit branches must reproduce parent at initialization; Ridge
is evaluated with its real nonzero initialization. All frozen weights must
remain unchanged and no parent parameter may accumulate gradients.

Actual launch: see LAUNCH.json. Source `d88495a`, immutable source_v5,
execution_v5, four simultaneous processes on physical GPU0. All four real
smokes passed after fixes; 15 tests passed in the actual cmerc-repro environment.
Earlier failed/invalid attempts are retained in INVALID_ATTEMPTS.json and do
not participate in ranking. The conditional DDPM has310401 trainable parameters
and its bridge265158; the unavailable archive's610753 denoiser count is not
reproduced or silently advertised. The20-step linear beta schedule has a
relatively high terminal alpha_bar, so Gaussian-prior approximation is a known
limitation of this declared first configuration, not a tuned guarantee.
