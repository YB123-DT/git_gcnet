# Supervised frozen modality Teacher: implementation verification

Engineering verification only. No complete Teacher pretraining or five-seed
Student experiment has been launched, and no F1 improvement is claimed.

## What changed

- `gcnet_missing_m3/pretrained_teacher.py`: validates complete emotion-only,
  validation-selected source checkpoint; strictly exports/loads only
  `observed_set.projectors.*`; validates keys, dimensions, dtypes, finite values
  and hashes. Does not load the Stage1 checkpoint's dormant `teacher.*`.
- `model.py`: default `teacher_mode=ema` is preserved. New `pretrained-frozen`
  mode loads Teacher only after all Student/MMoE initialization; Teacher stays
  eval/frozen; `update_teacher` explicitly becomes a no-op. No new parameters.
- `train_gcnet.py`: config/CLI, skips EMA for fixed targets, checks source
  dataset/fold/seed/latent dimension, checks Teacher state hash before training,
  after training before restoring checkpoints, and after evaluation.
- Independent stage runner, smoke and three test files. No loss/OSRAM scan edit.

## Stage protocol

**Teacher:** full0 inputs, emotion-only, original MSE on MOSI, complete validation
W-F1 checkpoint selection, no test evaluation. Reuse the original epoch budget as
a maximum; export the best validation checkpoint, not automatically the last.
Save actual epochs, updates and elapsed wall time as the additional Stage1 cost.

**Student:** fresh original initialization, cyclic rates, original contextual
MMoE, original source mean, SmoothL1/InfoNCE/temperature/weight. Only Teacher
projectors come from Stage1 and remain frozen. Predictions never feed emotion
inference. Retain per-seed/per-rate Test-oracle selection for this diagnostic
project; it is not a formal held-out paper result. Record Stage1 cost separately.

The current dedicated runner is MOSI/seed 66–70 only; other datasets are not
launched or claimed verified. Stage entrypoints are explicit and never chain
automatically. A diagnostic-only smoke Teacher is rejected by the full runner.

```bash
# Config-only inspection; does not train
python experiments/osram_supervised_teacher_20260914/run.py --stage teacher --seed 66 --check
python experiments/osram_supervised_teacher_20260914/run.py --stage student --seed 66 --check
```

Remove `--check` only for a separately authorized stage training run. Student
requires its matching Teacher export and COST.json to exist first.

## Real MOSI smoke (GPU 6)

Exactly one full-view Teacher training batch, then real complete validation;
one fixed-target Student training batch at miss=.5. This is not a trained
Teacher-quality experiment. The source checkpoint is marked diagnostic-only.

| Evidence | Result |
|---|---:|
| Stage1 emotion loss | 2.388966 |
| Audio projector emotion-gradient L2 | 1.565761 |
| Text projector emotion-gradient L2 | 2.178437 |
| Visual projector emotion-gradient L2 | 2.460852 |
| Stage1 online projectors changed | true |
| Stage1 stale EMA projectors unchanged | true |
| Stage2 Student/MMoE initialization matches baseline | exact |
| Stage2 emotion loss | 2.844248 |
| Original JEPA loss | 3.627940 |
| Regression branch loss | 0.442778 |
| Contrastive branch loss | 6.813102 |
| Total loss | 3.207042 |
| Missing targets | 1,142 |
| JEPA→predictor gradient L2 | 4.550469 |
| Stage2 EMA updates | 0 |
| New model parameters | 0 |

Teacher parameter/buffer hashes before/after are identical:
`d2cf7d15460c9d118d3a28c1ef954f64166854675f4ff78999e794af52dc145f`.
Teacher has no gradients. Both mode and manual EMA-call protections were
exercised. Test/inference works without Teacher target generation.
Raw evidence: [SMOKE.json](SMOKE.json).

## Tests and limitations

New tests cover poisoned stale EMA keys, exact Student initialization and RNG,
old state keys/strict restore, unchanged default EMA behavior, missing/extra
keys, wrong shapes, nonfinite weights, wrong selection/rate, exports/hashes,
frozen optimization, source metadata and stage-specific configuration.

Related regression suite: 351 passing tests after preserving the legacy
positional TrainConfig field order by appending new fields. Python compilation
and diff whitespace checks pass. Real smoke exits successfully with finite
losses/gradients. Model and trainer review found no blocker for this pipeline.

Limitations: the loader verifies metadata and tensor hashes, not the scientific
quality of a source checkpoint. Complete-model task training does not guarantee
every individual modality projector is sufficiently emotion-informative or
cross-modally predictable. This comparison changes both target initialization
and target update policy; it cannot uniquely attribute a gain to supervised
pretraining. Extra pretraining cost is not equivalent to the old one-stage budget.
No long-run stability, target-quality or performance result exists yet.
