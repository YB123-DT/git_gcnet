# cfg84 Flat + history-input scalar Gate: completed

INTERNAL TEST-ORACLE DIAGNOSTIC; NOT A FORMAL PAPER RESULT.

All three seeds66/67/68 completed100epochs on biggpu hostGPU0; child exitcodes0,
coordinator status complete. Source implementation237c03c; launch5bb617d.
Original Flat baseline was reused; no reference retraining. Same cfg84 cyclic
random-missing, no-JEPA, one-stage joint training from scratch.

| Metric W-F1 (%) | Flat | History-input Gate | Paired delta pp |
|---|---:|---:|---:|
|mean_8rate|80.559 ± 0.507|79.892 ± 0.673|-0.668 ± 0.238|
|high_missing|75.594 ± 1.016|75.239 ± 1.201|-0.355 ± 0.195|

SampleSD across three seed means. All three seeds decline on both aggregates.

| Seed | Flat8rate | Gate8rate | Delta pp |
|---|---:|---:|---:|
|66|81.068|80.427|-0.641|
|67|80.556|80.112|-0.444|
|68|80.053|79.136|-0.917|

| Missing rate | Flat mean | Gate mean | Delta pp |
|---|---:|---:|---:|
|0.0|88.419|87.060|-1.359|
|0.1|85.845|84.974|-0.871|
|0.2|83.431|82.571|-0.860|
|0.3|80.999|80.471|-0.528|
|0.4|78.996|78.338|-0.658|
|0.5|77.327|76.037|-1.290|
|0.6|75.848|75.772|-0.076|
|0.7|73.607|73.909|0.302|

## Selected-checkpoint alpha diagnostics

Ranges below are over eight selected test-rate checkpoints, not final-epoch
training statistics. Saturation is defined in implementation as alpha<=.81
or alpha>=1.19; 100% near-boundary does not mean exact mathematical tanh saturation.

| Seed | Alpha mean range | Within-evaluation alpha std range | Near-boundary fraction |
|---|---:|---:|---:|
|66|.823059–.919655|.030190–.086662|5.539%–45.481%|
|67|.800266–.801396|.000284–.000799|100%|
|68|.800096–.800229|.000121–.000311|100%|

Seeds67/68 choose alpha<1 for every evaluated valid utterance at all selected
checkpoints and show almost constant near-lower-bound behavior. Seed66 also
mostly weakens history (76.239%–99.854% of valid tokens). This implementation
therefore did not consistently learn the intended bidirectional sample-adaptive
behavior. This describes observed behavior, not a uniquely established cause
of performance decline. Finite-alpha oracle correction headroom does not imply
that end-to-end sentiment training can identify those oracle choices.

## Verification and artifacts

24 saved NPZ prediction files independently reproduce every recorded weighted
F1 exactly (nonzero labels only; strict >0 threshold). All histories have100
entries. Each completed run provenance records successful canonical evaluation
mask checks. No new inference or training was performed to collect this report.

Metrics/configs/history/provenance/predictions retained in results/. Checkpoints
remain at /data1/yb/remote_experiments/osram_cfg84_history_input_gate_20260929/runs
on biggpu; no weight files are committed. Earlier launch/status documents are
historical snapshots and are superseded by this completed result.
