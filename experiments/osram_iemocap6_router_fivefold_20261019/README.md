# IEMOCAP-6 five folds × three seeds: Flat / one-stage Router

INTERNAL EXPLORATORY EXPERIMENT — partial fold5 evidence is not five-fold evidence.

User approved completing five-fold, three-seed IEMOCAP-6 after MOSI on2026-10-19.
Reuse the six completed fold5 runs from
`../osram_text_path_router_3seed_20261019/RESULT.md`; add ONLY folds1–4 ×
seeds66/67/68 × Flat/Joint =24 fresh runs. Final scope is30 runs, not30 new runs.

## Fixed protocol

- Same current cfg84 causal no-JEPA Flat1600 and one-stage Text-source Router.
- `gcnet-random-mask-v1`, cyclic eight training rates0.0–0.7; no persistent mix.
- IEMOCAP original six-class CE; Router optimizes the corresponding mixed
  probability NLL. No auxiliary loss; same one-scan branch architecture.
- Adam, backbone LR1e-3, Router LR3e-4, constant schedule, weight_decay1e-5,
  batch32, original dropout, 100 epochs as in completed fold5.
- Each test fold is its actual held-out session. An independent conversation
  holdout is built within that fold's original training conversations using
  the SAME splitting rule as completed fold5. No fold5 masks/checkpoints
  substitute for other folds; test conversations never train or select.
- Per run: ONE checkpoint selected by validation eight-rate mean ACC, tie
  lower CE. UA and W-F1 are computed from the same selected predictions.
- Common split/eval seeds66; requested seed changes fresh initialization and
  training loader/RNG. Pair Flat/Router within each fold/seed.

## Runtime and ordering

Server: biggpu. Python: `/data2/yb/conda_envs/cmerc-repro/bin/python`;
Torch1.13.1+cu117, NumPy1.26.4, sklearn1.3.2. Original accepted feature/label
paths and hashes remain unchanged. DataLoader workers0.

The queue waits for the new six-job MOSI queue to finish successfully. It then
executes the24 approved jobs with bounded GPU concurrency and RAM checks.
No kill/retry/protocol change to squeeze jobs into a GPU; GPU4 is forbidden.
Every run keeps independent config/source/mask/label metadata and complete
model/Adam/RNG checkpoints. Existing fold5 outputs remain read-only.

## Aggregation

Report each fold/seed/rate ACC, UA, W-F1; pairwise initialization, training-mask
and evaluation-mask equality. Compute each seed's arithmetic mean over five
folds, then mean ± sample standard deviation over seeds. Also report mean8
and high missing0.5/0.6/0.7. Clearly distinguish fold-mean metrics from any
separately reported pooled-out-of-fold predictions; they are not identical.

## Ordered entry point

From the SAME immutable source snapshot as MOSI:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
  experiments.osram_iemocap6_router_fivefold_20261019.run \
  --output /data2/yb/remote_experiments/osram_iemocap6_router_fivefold_20261019/execution_v1 \
  --source-commit SOURCE_SHA \
  --gpus GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
         GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
         GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a \
  --after-queue /data2/yb/remote_experiments/osram_mosi_gcnet_router_3seed_20261019/execution_v1/QUEUE.json
```

This controller may be installed immediately under tmux, but performs no smoke
or training before the MOSI queue succeeds. Upstream failure blocks it rather
than silently changing the order. GPU7/1/3 carry at most2 training children each;
folds1–4 fill released slots. Completed fold5 source hashes are checked before use.
