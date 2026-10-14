# Frozen Query versus Decision — MELD seed66

INTERNAL DIAGNOSTIC ONLY

Completed rates: 8/8; status: complete.

Original CHM/Local/Flat are frozen. B fits a new small decoder; C freezes that selected B decoder and changes only Query.
Donors come from a different conversation in the same split, matched on current availability and history presence without labels.
Paired scores use the common donor-matched population; the full A reference is separate. Test masks replay the source winning epoch.
Each fit uses 30 epochs (epoch0 included), test-best ACC; associated UA/W-F1 are not separately selected.

| Model | 8-rate ACC | UA | W-F1 | High ACC | High UA | High W-F1 |
|---|---:|---:|---:|---:|---:|---:|
| A | 60.674 | 35.436 | 56.805 | 56.411 | 29.322 | 50.691 |
| B_local | 60.204 | 37.929 | 57.450 | 55.760 | 31.195 | 51.509 |
| B_donor | 59.064 | 36.367 | 56.179 | 54.049 | 28.997 | 49.630 |
| B_real | 59.294 | 37.640 | 56.907 | 54.342 | 30.975 | 50.741 |
| C_donor | 59.620 | 37.349 | 56.927 | 54.700 | 30.351 | 50.629 |
| C_real | 59.515 | 38.045 | 57.213 | 54.623 | 31.406 | 51.105 |

## Interpretation boundaries

- B_real versus B_local/B_donor tests usable history signal under this decoder class. B versus A also changes head capacity, regularization and fitting protocol.
- C_real versus its exact frozen B_real decoder tests whether this small Query correction can improve retrieval for that decoder. C_donor controls added current-conditioned function capacity.
- No gain does not prove Memory has no information or that Query is globally optimal. A gain on test-selected probes is not a deployable model result.
- No D raw-history fit, backbone training, auxiliary loss, multi-seed extension or architecture search was performed.

See PER_RATE.csv for per-rate scores, AGGREGATE.json for corrections/harms and current-availability splits.

See [INTERPRETATION.md](INTERPRETATION.md) for the evidence-bounded conclusion,
paired corrections/harms, full-versus-matched A distinction, gradient checks,
and measured run cost. This round did not establish a real-history advantage
beyond the Local control or isolate Query as the main bottleneck.
