# MOSI: three one-stage joint blocks

INTERNAL TEST-ORACLE SCREEN. One seed (66), finite experiment, no automatic HPO.

All four arms start from identical original random Flat parameters. No trained
checkpoint is loaded. The original encoder, OSRAM, Flat and task head remain
trainable. The fourth arm is an identically configured Flat control: Torch1.13
UUID-based CUDA RNG enumeration is repaired in every arm, unlike the historical
control, so the old scores alone are not an adequately matched comparison.

| Arm | New branch | Joint objective |
|---|---|---|
| Flat | None | Original MSE |
| NeuMiss | Observed coordinates + current context → masked iterative transform → pre-LN correction | Original MSE |
| Deep Signature | Strictly past eight valid utterances → per-modality degree-two signature → pre-LN correction | Original MSE |
| CSDI | Differentiable 20-step conditional denoising, four hypotheses → individual task predictions → mean prediction | Original MSE + noise MSE |

The modules reuse `osram_four_block_wiring`; the old frozen experiments are not
modified. Live hooks capture attached latents and the true pre-LN anchor in ONE
original CHM scan. No repeated Flat/Memory scan and no detached parent cache.
The zero-initialized exits preserve the original initial predictions. Their
upstream task gradients start after those exits update, not at the initial step.

CSDI has no pretraining phase and no frozen denoiser. Task gradients traverse all
reverse steps via checkpointing; noise MSE weight is fixed at 1 (unoptimized).
Noise-loss conditioning and complete TRAIN targets are detached, so that auxiliary
loss trains the denoiser directly; the task loss updates the entire parent and
denoiser. Complete features never enter test prediction. The fixed rank64 latent
PCA uses TRAIN features encoded by the INITIAL RANDOM parent, not a pretrained
parent. PCA fitting is preprocessing without emotion supervision, not a training
stage. Its buffers are saved and never refitted during optimization.

The inherited denoiser is our documented self-contained adaptation (310,401
parameters), not a claim to reproduce an unavailable 610,753-parameter reference.
Its inherited 20-step linear beta schedule 1e-4…0.02 has terminal alpha_bar about
0.817, hence imperfect terminal-noise matching is a known limitation. No silent
schedule tuning is included. Fixed test noise is keyed by rate/conversation/turn;
only standard noise is cached, never predictions or evolving representations.

## Fixed protocol

- biggpu, cmerc-repro Python/Torch1.13.1+cu117/NumPy1.26.4/sklearn1.3.2.
- MOSI official existing split/features, original gcnet-random-mask-v1,
  cyclic training rates 0–0.7, batch32, Adam, WD1e-5, seed66.
- 150 epochs: LR3e-4 first100, exact Python `3e-4/10` final50. No early stopping.
- Original MSE includes gold0. Original polarity ACC/W-F1 omit gold0.
- Reuse accepted Flat's exact eight TEST mask banks with metadata/order checks.
- Test W-F1 selects a separate best checkpoint for each rate, earliest tie.
  The eight-rate average is not a single-checkpoint deployment score.
- Report MSE, eight-rate ACC/W-F1, high-missing rates0.5/0.6/0.7, and same-model
  branch-on/off corrections/harms. Branch-off uses jointly trained weights;
  it is NOT the independent Flat control.
- All original writing, addressing, masking, loss and checkpoint artifacts stay
  intact. No independent new learning rates or auxiliary objectives except the
  specified CSDI denoising objective.

## Execution

Freeze the committed source in a new remote directory. From that snapshot:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_three_block_joint_mosi_20261025.scheduler \
  --root /data2/yb/remote_experiments/osram_three_block_joint_mosi_20261025/execution_v1 \
  --source-commit ACTUAL_SOURCE_SHA \
  --gpu0-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
  --gpu1-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
```

The persistent controller first runs three real full-size TRAIN batches (one
batch per epoch, cyclic masks) and eight-rate limited test replay in each arm.
Only after all smoke arms pass and measured peak memory fits does it launch the
four formal jobs. GPU0: Flat/NeuMiss/Signature. GPU1: CSDI. GPU4 is forbidden.
No smoke scores are formal results. A failed arm blocks formal launch, with logs
retained. No automatic hyperparameter changes or unbounded retries.

Each run saves identity/source hashes/environment/data provenance/config,
`history.json`, exact `fixed_masks.pt`, selected checkpoint/prediction hashes,
`last.pt` (model/Adam/epoch/RNG/dedicated noise-generator state), `SUMMARY.json`,
and `RESULT.json`. Use `run --resume` only against the identical identity and
immutable snapshot. Test-time keyed noise is reconstructible from its keys.
The controller records PIDs and commands in QUEUE.json and summarizes all arms.
Source and logs are evidence of implementation/execution, not evidence of gain.
