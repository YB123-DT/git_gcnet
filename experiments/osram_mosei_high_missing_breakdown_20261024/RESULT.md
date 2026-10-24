# MOSEI high-missing availability breakdown

INTERNAL TEST-ORACLE SCREEN. Frozen saved predictions only; no training or inference.

Source: `/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none`.
Original Flat1600 seed66, cmerc-repro, LR1e-4 constant, 60epochs,
gcnet-random-mask-v1. Uses each rate's exact selected checkpoint predictions
and saved masks, NOT the historical unrepaired-RNG three-seed Flat results.
Rates .5/.6 select epoch16, .7 selects epoch23 by test W-F1.

## Current availability: W-F1 (%)

| Current availability | miss .5 | miss .6 | miss .7 |
|---|---:|---:|---:|
| A | 74.406 | 75.408 | 73.864 |
| T | 87.091 | 86.539 | 86.365 |
| V | 78.782 | 74.503 | 73.863 |
| AT | 84.747 | 86.117 | N/A |
| AV | 78.660 | 74.595 | N/A |
| TV | 89.087 | 84.793 | N/A |
| ATV | 86.940 | 92.056 | N/A |
| Overall | 81.961 | 79.700 | 78.035 |

N/A means no samples, not zero score. At .6 ATV has only38 nonneutral
samples; do not infer an advantage from this small subgroup.

## miss .7 polarity sample counts

| Availability | Nonneutral N | ACC | W-F1 | Errors |
|---|---:|---:|---:|---:|
| A | 1226 | 74.470 | 73.864 | 313 |
| T | 1187 | 86.352 | 86.365 | 162 |
| V | 1221 | 74.447 | 73.863 | 312 |
| Overall | 3634 | 78.343 | 78.035 | 787 |

Current no-Text groups account for2447/3634=67.336% of polarity samples
and625/787=79.416% of errors. All current groups still use the original
full Memory model; these are NOT Persistent Missing experiments.

## Current Text and strictly past Text exposure: W-F1 (%)

| Group | miss .5 | miss .6 | miss .7 |
|---|---:|---:|---:|
| Current Text present | 87.023 | 86.360 | 86.365 |
| Current Text missing, history has Text | 78.203 | 76.490 | 75.301 |
| Current Text missing, history never has Text | 68.893 | 66.458 | 69.654 |

At .7, nonneutral group N respectively1187/1618/829. History exposure is
computed strictly before the current utterance; no cross-dialogue accumulation.
These are observational partitions: different utterances/lengths/label balances
and different per-rate checkpoints/masks prevent a causal interpretation of
the score gaps. In particular, this does not establish the benefit of Memory
or isolate write/query/readout failure. W-F1 subgroup averages do not add up
to overall W-F1; ACC/error counts do aggregate by support.

## Validation and reproduction

`analyze.py` checked all eight selected prediction SHA256s and mask SHA256s,
binary nonempty availability, finite predictions, contiguous per-conversation
utterance indices, exhaustive partitions, error/count reconciliation, and
reproduced all eight published ACC/W-F1/MSE to1e-5 tolerance. At .7 every
valid utterance has exactly one observed modality. ACC/W-F1 exclude neutral
labels, use positive iff prediction>0; MSE includes neutral labels.

Executed with `/data2/yb/conda_envs/cmerc-repro/bin/python` on biggpu:

```bash
python analyze.py \
  --source /data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66/none \
  --output /data2/yb/remote_experiments/osram_mosei_high_missing_breakdown_20261024/execution_v1
```

Complete eight-rate subgroup metrics, counts, errors, MSE and provenance hashes:
`RESULT.json`. No GPU allocated; no running experiment, checkpoint, or source
artifact modified. Single seed, exploratory test-selected results only.
