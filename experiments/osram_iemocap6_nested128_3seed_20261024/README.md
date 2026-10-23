# Fixed IEMOCAP-6 Nested128 — five folds, three seeds

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.

User promoted the seed66/fold5 `nested_dim128` result (ACC65.689, UA63.895,
W-F1 65.748). Stop other IEMOCAP6 variants; do not add IEMOCAP4, HPO or Flat
training. Existing MOSEI workers are untouched. No model/sampler math changes.

Exactly15 seed/fold results: seeds66/67/68 × folds1–5. Reuse the verified complete
seed66/fold5 trajectory read-only; train14 new independent trajectories. Every
new run starts from the canonical seeded random initialization, NOT old weights.
Model parameters13,950,441, including432,387 Nested parameters. Graph dimension
128, depth3,8groups and the existing default GIN hidden128; original residual
and zero-initialized decoders remain unchanged.

- cmerc-repro Python3.10/Torch1.13.1+cu117/NumPy1.26.4/sklearn1.3.2.
- Original cfg84 no-JEPA Flat1600, causal CHM8×64, write_step0.6, LocalSkip.
- One-stage end-to-end original CE, Adam LR1e-3 constant, WD1e-5, batch32,100epochs.
- Original `gcnet-random-mask-v1` cyclic0.0–0.7 training; host/guest/padding unchanged.
- Every fold uses its accepted Flat66 rich selected test-mask artifacts; all
  three training seeds share this same fold-specific test bank and seed66 split.
- Train loader/model/training stochasticity use the ACTUAL training seed; factory
  uses the ACTUAL fold. Official LOSO training uses four sessions with no new holdout.
- Per-rate Test-best ACC, strict improvement/earliest tie; UA/W-F1/CE come from
  the same predictions/mask. Eight BEST plus exact predictions/masks and full
  model/Adam/CPU+CUDA RNG recovery state are retained. No validation selection.

The shared experiment worker gains seed/fold arguments ONLY for IEMOCAP6 Nested128.
Its default66/fold5 configuration is unchanged (regression tested). All numerical
`gcnet_missing_m3` and `gcnet_modality_jepa` sources remain identical to the reused
run. Old running immutable snapshots are not edited. A complete-source manifest
binds every newly admitted run; independent output paths prevent overwrite.

## GPU0 and six-process request

biggpu GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`; GPU4 forbidden.
The full100-epoch reference measured peak5536MiB (not just its short smoke).
Six such peaks alone need33,216MiB, exceeding the card's32,768MiB even before
driver/context margin. Therefore the requested ceiling is6, but reservations
of6000MiB per worker plus1250MiB global margin safely admit at most5 on this card.
External allocations are charged, own usage is not double-counted, and usage
above the reservation stays charged. No batch/model/missing/precision change
or killing unrelated jobs is used to force concurrency.64GiB host RAM and25GiB
disk minimums apply. Pending jobs automatically fill safe released slots.

Initial order exposes all three seeds quickly:66/fold1,67/fold5,68/fold5,
66/fold2,67/fold1,68/fold1; remaining pairs are deterministic. The already
completed66/fold5 is never resubmitted. Failures drain active peers and block
pending jobs, without silent OOM retries or altered hyperparameters.

Run from a committed immutable archive under persistent tmux:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_iemocap6_nested128_3seed_20261024.run \
  --source-commit SOURCE_SHA --output /absolute/fresh/execution_v1 \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 --parallel 6
```

## Interpretation and outputs

The current selected model is exploratory: seed66/fold5 helped select it.
Full three-seed results appear only after all15 trajectories pass integrity
checks. Fivefold means equally weight folds, as the existing protocol; sample
SD is computed across the three complete fivefold means. Report all eight rates,
high0.5/0.6/0.7, each seed and each fold in RESULT.md/json and two CSVs.

Do NOT present historical Flat3seed scores as strict paired differences: those
old runs resampled evaluation masks by epoch and had a different treatment of
fold-specific initialization. This queue does NOT secretly add15 new Flat runs.
The previously computed same-source66/fold5 Flat remains its paired reference;
the rest of historical Flat results are context only. No mechanism/success claim
follows merely from completion. No automatic extra folds/seeds/search follows.

All old BEST weights/records are protected. The interrupted old IEMOCAP6 groups4
run has its explicit user-cancellation receipt in the earlier queue directory;
MOSEI workers continue, even though that old controller records the interruption
as failure-draining. Do not interpret that cancellation as a new Nested128 failure.
