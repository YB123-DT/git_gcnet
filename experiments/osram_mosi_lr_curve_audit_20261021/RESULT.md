# MOSI learning-rate and V3 training curves

INTERNAL TEST-ORACLE DIAGNOSTIC. Seed66 only; descriptive, no significance
claim. All three150-epoch runs were completed before this analysis. No new
training, inference, checkpoint loading or running-task changes were made.

## Evidence and reproducibility

`curves.json` contains450 flattened historical epoch records, exact remote
source paths and SHA256 of each original history and summary. All150 training
mask traces, valid-sample counts and update counts match across the three
runs. Each epoch has2 optimizer updates; all three perform300 updates total.
Original data, GCNet-random-mask-v1, batch32 and task MSE remain matched.

- Flat originalLR:1e-3 through epoch100, then1e-4.
- Flat lowerLR:3e-4 through epoch100, then3e-5.
- V3 lowerLR:3e-4 through epoch100, then3e-5.

Run locally with existing NumPy/matplotlib:

```bash
python experiments/osram_mosi_lr_curve_audit_20261021/plot.py
```

This validates completeness, finite values, update counts and LR transitions,
then generates `curves.svg`, `curves.pdf`, `curves.csv`, `window_summary.json`.
The PDF preview was visually checked for readable labels, unclipped content,
distinct methods and the epoch101 transition marker. These are exploratory
log plots, not publication-layout or screenshot-to-reference fidelity work.

## Reading the four panels

- A: training MSE; raw values and N-weighted trailing8-epoch mean, logarithmic
  MSE axis. Cyclic training samples rates `(epoch_zero_based+batch_index)%8`,
  so adjacent epochs differ in mask conditions. Raw oscillations alone are not
  a learning-rate instability measurement.
- B: test MSE averaged over all8 rates from the SAME checkpoint at each epoch;
  raw and trailing8-epoch mean, log axis.
- C: corresponding same-checkpoint test8-rate W-F1, not per-rate-best scores.
- D: cumulative per-rate TEST-best W-F1 averaged across8 rates. Different
  rates may select different epochs. This panel is a selection envelope, NOT
  one checkpoint or a directly deployable score. It intentionally plateaus
  when later epochs do not improve any previously selected rate.

## What the existing curves actually show

| Run | Train MSE epochs41–48 | Train MSE137–144 | Test W-F1 epochs41–48 | Test W-F1137–144 |
|---|---:|---:|---:|---:|
| Flat1e-3→1e-4 | 1.6241 | 0.6618 | 70.0263 | 75.4809 |
| Flat3e-4→3e-5 | 1.1009 | 0.4742 | 76.0313 | 74.0933 |
| V3 3e-4→3e-5 | 0.9835 | 0.4843 | 75.9391 | 74.8769 |

Window values average exactly8 consecutive epochs; training means are
N-weighted. Test windows average the same-epoch8-rate means, NOT an oracle
selection across window epochs.

1. In this seed, smallerLR Flat reaches lower training MSE and stronger test
   performance earlier. Original1e-3 has a pronounced early test-performance
   trough and catches up later. This is stronger evidence than comparing
   best epochs alone, but is not a universal claim that largerLR converges
   slower or proof that all variation is caused by LR-driven oscillation.
2. Epoch101 decay helps the originalLR run's later test performance: test
   W-F1 averages76.2736 over89–96 and77.4785 over105–112. The same-checkpoint
   best occurs at103(78.1822), versus48(78.6910) for lowerLR Flat.
3. All training-loss curves continue decreasing late, without a corresponding
   monotonic test W-F1 improvement. Therefore early test-best does not establish
   completed optimization convergence; nor does more fitting ensure better
   held-out predictions. No overfitting mechanism is proven by these logs alone.
4. V3's trajectory is not identical to Flat, even though final per-rate-best
   means are close: originalLR Flat79.1280, lowerLR Flat79.2491, V3 79.1394.
   V3's same-checkpoint best is epoch75(78.3482). These are different summaries:
   do not compare a same-checkpoint curve point to a per-rate-best score as if
   they were the same metric.

No learning-rate change, early-stop rule, new trial or seed expansion is
authorized or performed by this retrospective curve inspection.
