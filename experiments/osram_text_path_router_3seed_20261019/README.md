# One-stage Text-source Router: fixed three-seed extension

Reuse completed seed66 Flat/Joint runs; train only seeds67/68 for MELD and
IEMOCAP-6 fold5, eight new jobs total. This is not a new algorithm or a parameter
search. No MOSI jobs are submitted by this launcher.

The model, probability-mixture CE, one-scan shared-dropout readouts, and all
Memory/query/write definitions are unchanged. Each seed initializes a fresh
backbone and Router. Flat/Joint use the same initialization and training order
within each seed. The canonical seed-derived training loader and training RNG
vary with seed; the original seed66 conversation holdout, validation mask bank,
and exact saved test masks remain fixed for paired comparison across seeds.

MELD: 50 epochs, backbone LR1e-5, wd1e-5, batch32. IEMOCAP-6 fold5: 100 epochs,
backbone LR1e-3, wd1e-5, batch32. Joint Router LR3e-4. One shared validation
eight-rate ACC checkpoint is selected per run; no per-rate Test-best selection.

Run from an immutable source snapshot on `ssh biggpu`:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -m experiments.osram_text_path_router_3seed_20261019.run \
  --source-commit SOURCE_SHA --output /home/qinjinghui/remote_experiments/osram_text_path_router_3seed_20261019/execution_v1 \
  --gpus MELD_HEALTHY_GPU_UUID IEMOCAP_HEALTHY_GPU_UUID
```

The launcher verifies completed seed66 references at
`/home/qinjinghui/remote_experiments/osram_text_path_router_joint_20261019/execution_v1/runs`,
unchanged core source hashes, and matching paired evaluation protocol. It rejects
physical GPU4 and its known UUID, checks observed training peaks with a 1.5x
factor plus 2GiB margin, then starts four jobs per card. No additional mathematical
smoke is needed after the unchanged algorithm's seed66 GPU checks. Insufficient
capacity fails before submission instead of changing batch size or moving jobs.

`QUEUE.json` persists commands/PIDs/status/exits. Each new child writes to
`runs/{MELD,IEMOCAPSix}/seed_{67,68}/{flat,joint}` with its own log, complete
best/last recovery checkpoint, source/config hashes, training mask trace hashes,
fixed validation/test mask hashes, predictions and metrics. The final three-seed
report runs automatically after all eight succeed. Failures are not resubmitted,
and other jobs are never killed. Launch inside tmux to survive SSH disconnects.

Seed66 source differs only in the runner's seed/order/provenance support; its
model/source-tracker/Router computation is verified unchanged. The historical
protected test-oracle anchor is used only as an environment/parity gate, not as
training initialization. Historical exploratory/test-mask provenance remains a
limitation; this extension is not an independent confirmatory paper benchmark.
