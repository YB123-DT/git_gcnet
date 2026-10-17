# Execution and interpretation

Offline analysis completed2026-10-17 on biggpu. No model training, checkpoint
loading, inference or GPU allocation; only existing NPZ/JSON artifacts read.
Historical RESULT.md, predictions, features and model files were not changed.

The actual source root has two rate-directory levels:
`full_68c698f_v2/rate_0p7/rate_0p7/probes/seed_66/...`.
Each outer rate extraction has a complete STATUS.json. All four relevant
prediction files were present for8 rates and3 probe seeds.

Run in `/data2/yb/conda_envs/cmerc-repro/bin/python`:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
/data2/yb/conda_envs/cmerc-repro/bin/python correctness_cross.py \
  --input /data2/yb/remote_experiments/osram_meld_information_availability_20261014/full_68c698f_v2 \
  --output /data2/yb/remote_experiments/osram_meld_information_availability_20261014/correctness_cross_20261017/results_v1
```

The executed script is copied under the new remote `correctness_cross_20261017`
directory; its SHA256 and every input artifact hash are in CROSS_TABLES.json.
The repository source is one level above this report directory. The script
rejects an existing output directory; reproduction must use a new output path.

Four new tests first failed for the absent implementation, then8 new/existing
analysis tests passed. A separate artifact check validated72 unique
rate/probe-seed/comparison rows, all four-cell totals, the exact
`delta_ACC=(corrections-harms)/N` identity, coverage2608–2610, and equality of
the executed-script hash and the repository script.

## What these results do and do not show

For seed66/missing0.7, Local gets1401/2610 correct and1209 wrong.
Base gets161 of those1209 errors correct (13.317%); Memory gets100 (8.271%).
Thus memory-derived representations support some different correct predictions.
This is not a usable label-free selector and is not proof of unused signal in
the current gcnet-random-mask-v1 model.

The learned combined LocalMemory probe corrects117 Local errors but loses124
Local-correct samples: net-7 correct, ΔACC-0.268pp. On this same cohort its UA
increases1.381pp and W-F1 increases1.153pp, again showing the ACC/UA tradeoff.

Across all8 rates, the paired LocalMemory-minus-Local mean ACC deltas are:

| Probe initialization seed | Mean corrections/rate | Mean harms/rate | Mean ΔACC pp |
|---|---:|---:|---:|
|66|102.000|113.250|-0.431|
|67|93.375|110.875|-0.671|
|68|88.625|105.875|-0.661|

All three probe initializations have negative eight-rate net ACC gain.
The three-seed average paired delta is-0.588pp. These are three probes fitted
on one frozen backbone seed66, not three independent backbone training runs.

Base/Memory include history-presence/count metadata and use independently fitted
heads. Consequently Base-versus-Memory correctness differences cannot be called
the causal incremental effect of Gap. Likewise Local/LocalMemory is a probe
comparison, not same-checkpoint Memory-on/off intervention. No new bottleneck,
Gate, readout or performance claim is justified from complementarity alone.

INTERNAL DIAGNOSTIC ONLY: old gcnet-legacy-fixed-v1 masks, Torch2.2.2+cu121
probe-training provenance and Test-oracle selections. Do not substitute these
numbers for the current random-mask/cmerc-repro baseline.
