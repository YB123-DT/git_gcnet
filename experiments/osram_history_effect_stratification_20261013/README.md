# P0 frozen history-effect stratification

This internal diagnostic reuses the existing IEMOCAP and MELD Flat source
combination evaluators. It does not train or modify models. One original memory
scan per batch supplies captured Flat inputs for L, LB, and LBG. L contains the
Local Skip and adapter Local slice; B adds Base; G adds the three masked Gap
slots. LBG uses original forward logits after checking the complete replay
within the established `rtol=atol=2e-5` tolerance. Original saved predictions
are checked when available (required for IEMOCAP). Source checkpoints retain
their historical per-rate Test-oracle selection; these are descriptive
diagnostics, not independently trained ablations or unbiased model selection.

Example (use the source run's original data files and environment):

```bash
python -m experiments.osram_history_effect_stratification_20261013.evaluate \
  --source /path/to/source/seed_66/fold_1 \
  --dataset-root /path/to/datasets --device cuda \
  --output /path/to/new/p0/iemocap4/seed_66/fold_1
```

MELD additionally requires `--meld-pickle /path/to/source/MELD_features.pkl`.
The evaluator supports both IEMOCAPFour and IEMOCAPSix, and all nine MELD
speaker IDs. Historical MELD `gcnet-legacy-fixed-v1` masks are reconstructed
with the existing frozen legacy schedule. `--rates 0.0 0.5` restricts the
default rates 0.0--0.7. The output directory must not exist; no source or prior
output is overwritten.

Each `utterances_miss_*.npz` is readable with `allow_pickle=False`. Rows follow
the existing metric's batch-major valid-utterance order. It includes labels,
L/LB/LBG logits, predictions, CE, true-class probabilities, paired corrections
and harms, and the following metadata:

- Conversation ID, zero-based position, conversation length, numeric speaker.
- Strict prior-total and prior-same-speaker counts; speaker switch is -1 for
  the first utterance, 0 for the same adjacent speaker, 1 for a switch.
- Strict-past global and same-speaker modality-seen vectors in A/T/V order;
  current availability is excluded from both history vectors.
- Current availability in A/T/V order, adjacent previous-label relation
  (`start`/`same`/`different`), and previous same-speaker label relation.
  Label relations use ground truth for post-hoc analysis only.

`Delta_Base = CE(L) - CE(LB)` and `Delta_Gap = CE(LB) - CE(LBG)`; positive values
mean lower loss. `Delta_Total` is their sum. Probability change is after minus
before true-class probability. A correction changes wrong to correct; a harm
changes correct to wrong. `SUMMARY.json` reports counts, mean/std CE changes,
positive/negative/zero counts, corrections/harms and conditional rates,
probability change, and classification metrics for each rate. Strata include
history bins (0, 1, 2--4, 5--9, 10--19, 20+), speaker switches, availability,
modality seen, label relations, and availability crossed with prior total or
history seen. Every grouping partitions all rows. Standard deviations describe
utterances; correlated utterances are not treated as independent for inference.

Config and checkpoint hashes, original epoch, source mask protocol, device,
dataset identity/seed/fold, and MELD pickle hash accompany each run. Frozen
inference can require the historical runtime for source-exact reproduction.
Preserve all seeds and folds when aggregating; do not select the best strata.

## Dispatch and aggregate

The dispatcher plans 30 IEMOCAP sources (two datasets × three seeds × five
folds) plus three MELD sources. It validates source identities and required
files before launching, limits concurrency, writes separate logs, and records
commands/process exit statuses in `DISPATCH.json`. Select `--workers` according
to measured GPU headroom and throughput, and set `CUDA_VISIBLE_DEVICES` to a
healthy GPU whitelist on the source model's assigned server. It never migrates
models or selects GPUs automatically. All sources for a dataset must be on
their assigned server. Output roots must be new.

```bash
python -m experiments.osram_history_effect_stratification_20261013.dispatch \
  --iemocap-source-root /path/to/retained/iemocap_runs \
  --meld-source-root /path/to/retained/meld_runs/runs \
  --dataset-root /path/to/datasets --meld-pickle /path/to/source/MELD_features.pkl \
  --output-root /path/to/new/p0 --workers 3 --device cuda
python -m experiments.osram_history_effect_stratification_20261013.summarize \
  --input-root /path/to/new/p0 --output /path/to/new/p0_aggregate
```

Use `--datasets iemocap4 iemocap6` or `--datasets meld` for separate servers.
IEMOCAP roots contain `iemocap4/seed_66/fold_1` and
`iemocap6/seed_66/fold_1`; MELD roots contain `flat_seed66` through `flat_seed68`.
`--seeds` and `--rates` explicitly narrow the queue.

Aggregation requires complete run summaries and all five IEMOCAP folds for
each dataset/seed/rate. It recomputes effects from saved logits, rejects
duplicate runs and overlapping fold utterances, pools folds within each seed,
and then reports seed means/sample standard deviations. Rates and datasets
remain separate. Strata missing from some seeds retain explicit contributing
seed IDs and counts. `--allow-partial` explicitly labels incomplete fold
coverage; it is intended for early inspection only. Outputs are
`AGGREGATE.json`, `REPORT.md`, and `STRATA.csv`, including per-seed strata and
across-seed strata effects. Per-fold strata remain in the original run
summaries. The same three files are also written separately under
`IEMOCAPFour/`, `IEMOCAPSix/`, and `MELD/` for the datasets present. A new
directory is required for every aggregation.
