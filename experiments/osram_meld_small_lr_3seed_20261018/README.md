# MELD original Flat: fixed LR1e-5 / weight_decay2e-4, three seeds

INTERNAL TEST-ORACLE SCREEN. No further hyperparameter search or new Block.

Reuse completed O6 seed66 (ACC61.786398 / UA38.434973 / W-F1 58.917567).
Train only seeds67/68 from scratch, each50epochs, batch32, Adam, constant LR.
This is three total seeds, not three additional seeds. All remaining config
fields are copied exactly from the winning seed66 configuration.

Preserve original ECERC pickle SHA256
`a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`,
gcnet-random-mask-v1 host/guest padded-rectangle sampling, cyclic .0–.7,
Flat1600 and original Local Skip, emotion-only CE, causal8-head CHM with
write_step.6 and original joint Block Write. No second Block or extra loss.

Use unchanged b4a3138 numerical checkout and cmerc-repro environment as O6:
`/data2/yb/conda_envs/cmerc-repro/bin/python`, Torch1.13.1+cu117,
NumPy1.26.4, sklearn1.3.2. Shared helper hashes are checked against the original
optimizer screen manifest; this extension has its own source manifest.

Seed changes initialization, training order and random masks. These are intended
independent seeds, NOT identical-mask pairs with seed66. Each seed records its
actual full mask stream; do not fabricate cross-seed mask equality. No baseline
or other module is retrained in this task.

Each rate is selected by test ACC; UA/W-F1 are computed from the same selected
epoch/mask/predictions. BEST weights, exact selected predictions and complete
last_training state (optimizer/RNG/progress) are preserved. The reused observer's
`W0` label denotes independent initialization only; it does not load seed66's
weights, optimizer, predictions, or masks.

## Finite launch

On biggpu healthy physical GPU1, bind its actual UUID. Two new workers run in
parallel on that card, no changes to batch size. Existing O6 peak memory is reused
for admission; no redundant smoke or baseline training is added. GPU4 forbidden.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_small_lr_3seed_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_meld_small_lr_3seed_20261018/runs_v1
```

Launch from the frozen helper snapshot with the shared root SOURCE_MANIFEST.json.
The persistent controller checks O6 completion, starts S67/S68, and automatically
summarizes all three seeds as RESULTS.json and RESULT.md (mean and sample std,
overall/high-missing; individual summaries retain all eight rate results).
Use --resume only for the same immutable source/config/output; do not duplicate
an active controller. Initial config tests and original observer regressions:

```bash
python -m pytest tests/test_meld_small_lr_three_seed.py tests/test_meld_optimizer_screen.py -q
```

This follow-up checks cross-seed stability after seed66 test-set selection.
It does not remove the original screen's selection bias or establish formal
independent-test performance. No automatic additional seeds or search follows.
