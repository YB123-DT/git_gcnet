# Complete original Flat three-seed results in cmerc-repro

This finite queue trains only Flat seeds67/68 for IEMOCAPFour and IEMOCAPSix,
five folds each,100epochs:20 new fold jobs. The ten completed Flat seed66
outputs are reused read-only from:

```
/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016
```

No seed66 or B6 job is submitted. Completed B6 results remain the reference at
`/home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015`;
the launcher records this reference path without modifying or rerunning it.

The existing original-Flat builder now accepts an explicit seed subset.
Its default remains seed66, preserving the previous launcher behavior. New
seeds67/68 read their own original corrected-mask configs from the original
r1 baseline root. All original fields are retained, with only the established
bookkeeping changes: selection artifacts enabled, explicit ACC selection,
logical `cuda:0` after UUID binding, and neutral `osram_rpca_lr_multiplier=1`.
The meaningful block remains `none`; no B6, Nested, Gate or JEPA component is
enabled. Discarded historical weights are not loaded.

## Runtime and existing-result acceptance

The launcher requires the resolved executable
`/data2/yb/conda_envs/cmerc-repro/bin/python` and Torch1.13.1+cu117 before
preparation or launch. Every worker receives both `--flat-only` and
`--flat-replication`; the updated worker also requires CUDA11.7,
NumPy1.26.4 and scikit-learn1.3.2. New completed results must record this
runtime under `provenance.replication_environment`.

Seed66 reuse requires all ten complete `Original_Flat` summaries, exact
expected effective configs/dataset/seed/fold, sourceSHA785a012, valid finite
eight-rate metrics, and first-epoch retention checks with8 BEST weights,
8 selected artifacts and1 CUDA RNG state. Each summary must record
Torch1.13.1+cu117, while the old manifest must record the cmerc-repro
executable and sourceSHA785a012. Manifest, summary and retention hashes are
saved as immutable evidence in the new manifest. Existing outputs are neither
copied nor edited.

Numerical code remains frozen at:

```
/home/qinjinghui/remote_experiments/osram_iemocap_b6_source_785a012
785a01242ef1a82b97d5337acbb33ba963df5537
```

## Prepare and run

Use the new full immutable implementation checkout, which contains the
updated worker and both prior queue helpers. Numerical source remains the
separate785a012 checkout.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python \
  -m experiments.osram_iemocap_flat_3seed_cmerc_20261016.run \
  --worker /absolute/new-implementation/experiments/osram_iemocap_b6_transfer_20261015/worker.py \
  --smoke-summary /absolute/existing-B6-smoke/SUMMARY.json \
  --data-provenance /absolute/DATA_PROVENANCE.json \
  --check-only
```

`--check-only` validates the runtime, frozen source and all ten completed
seed66 references, then writes the immutable new manifest and20 configs.
It creates no GPU scheduler and launches no workers. Remove this flag for
the authorized persistent queue. Optional `--seed66-root`, `--baseline-root`,
`--source-root` and `--output` locate the recorded inputs/outputs. The new
default output is:

```
/home/qinjinghui/remote_experiments/osram_iemocap_flat_3seed_cmerc_20261016
```

Run under the project's persistent tmux/process mechanism and retain the
controller command, PID and log path. The inherited controller/scheduler
locks, immutable job checks, PID birth identity, UUID pinning and complete
RNG/checkpoint recovery prevent duplicate submissions and mixed protocols.
Changes to configs, helpers, worker or reused evidence reject resume in the
same output directory.

Only physical GPU1 and GPU3 are eligible, with at most4 and2 jobs from this
queue, respectively. Actual memory/process/disk admission can reduce these
limits. GPUs4/5 are excluded. The existing measured longest-batch B6 smoke
provides a conservative Flat memory bound using the larger allocated/reserved
peak plus1024MiB headroom; no extra GPU smoke is started. Independent active
jobs also reserve their measured peaks. External jobs are never signalled.

## Results and retention

After workers complete, `SUMMARY.json`, `SUMMARY.csv` and `RESULT.md` join
the20 new results with the10 reused seed66 results. A per-seed fivefold mean
requires all five valid folds; partial fold sets are never averaged. Three-seed
mean and sample standard deviation appear only after all30 valid fold runs
complete. Summaries retain per-rate, eight-rate mean and high0.5/0.6/0.7
ACC/UA/W-F1/CE. Failed or invalid jobs are excluded from aggregates.

All BEST weights, exact-selection-epoch artifacts and complete last-training
states are retained. Test-best results use independent ACC-selected epochs
at each missing rate and remain internal test-oracle diagnostics. Exact UA
uses the same selected logits/masks. No new result is claimed before actual
worker summaries pass the acceptance checks.

## CPU verification

```bash
python -m pytest -q tests/test_iemocap_flat_3seed_cmerc.py tests/test_iemocap_flat_seed66_restore.py
python -m py_compile experiments/osram_iemocap_flat_3seed_cmerc_20261016/run.py
```

Tests cover configurable seed subsets with unchanged seed66 defaults,
exactly20 new jobs, explicit replication flags, cmerc runtime rejection,
read-only complete seed66 validation, worker provenance-schema compatibility,
and30-result aggregation without partial three-seed averages. No GPU task is
started by these tests.
