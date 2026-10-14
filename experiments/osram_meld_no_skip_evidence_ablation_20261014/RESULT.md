# MELD retrained no-skip Base/Gap readout ablation

**INTERNAL TEST-ORACLE DIAGNOSTIC ONLY.** This is frozen inference on the
retrained seed66 no-Local-shortcut model. Every batch runs the unchanged causal
Memory exactly once; only captured Base and/or Gap slots are zeroed before the
same trained Adapter, LayerNorm, and classifier. Adapter Local is retained in
all four conditions.

- `LBG`: full no-skip model.
- `LG`: Base removed.
- `LB`: all Gap slots removed.
- `L`: Base and all Gap slots removed.

## Accuracy (%)

| Condition | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 8-rate mean | High mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LBG | 66.667 | 65.134 | 63.831 | 61.188 | 59.732 | 57.318 | 56.169 | 54.598 | **60.580** | **56.028** |
| LG (no Base) | 66.130 | 64.521 | 63.372 | 60.958 | 59.923 | 57.050 | 55.517 | 54.483 | 60.244 | 55.683 |
| LB (no Gap) | 66.667 | 64.483 | 63.525 | 60.766 | 59.655 | 57.203 | 56.130 | 54.598 | 60.378 | 55.977 |
| L (no Base/Gap) | 66.130 | 63.946 | 63.218 | 60.307 | 59.847 | 56.705 | 55.364 | 54.330 | 59.981 | 55.466 |

## Weighted F1 (%)

| Condition | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 8-rate mean | High mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LBG | 63.735 | 62.359 | 60.217 | 57.830 | 55.817 | 52.258 | 50.362 | 47.688 | **56.283** | **50.103** |
| LG (no Base) | 63.216 | 61.758 | 59.704 | 57.290 | 55.842 | 51.717 | 49.145 | 46.973 | 55.706 | 49.278 |
| LB (no Gap) | 63.735 | 61.825 | 59.858 | 57.470 | 55.639 | 51.942 | 50.147 | 47.480 | 56.012 | 49.856 |
| L (no Base/Gap) | 63.216 | 61.263 | 59.517 | 56.794 | 55.690 | 51.189 | 48.840 | 46.673 | 55.398 | 48.901 |

## Paired changes against Full

Counts pool the eight repeated rate evaluations.

| Intervention | Corrections | Harms | Changed predictions | Mean ACC delta | Mean W-F1 delta |
|---|---:|---:|---:|---:|---:|
| Remove Base | 289 | 359 | 936 | -0.335 pp | -0.578 pp |
| Remove Gap | 98 | 140 | 347 | -0.201 pp | -0.271 pp |
| Remove Base and Gap | 318 | 443 | 1,109 | -0.599 pp | -0.886 pp |

At high missing rates, removing Base changes W-F1 by `-0.824 pp`, removing Gap
by `-0.247 pp`, and removing both by `-1.202 pp`. Thus the retrained no-skip
model does exploit both historical evidence groups, with the larger aggregate
contribution coming from Base. The effect remains modest relative to the Local
information retained inside the Adapter. Individual examples can still improve
when evidence is removed, so this does not establish that every Memory read is
helpful.

Full replay exactly matches the original logits (`max_abs_error=0`) across 72
test batches. Source training commit: `60903a6`; evaluator commit: `bedc251`.
Remote artifacts:
`/data2/yb/remote_experiments/osram_meld_no_skip_evidence_ablation_20261014/seed66_bedc251`.
