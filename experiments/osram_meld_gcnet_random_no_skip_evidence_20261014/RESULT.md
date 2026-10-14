# Base/Gap contribution after removing the final Local shortcut

**INTERNAL DIAGNOSTIC ONLY.**

This is frozen, paired inference on the seed-66 MELD no-Local-skip model
trained with `gcnet-random-mask-v1`. Each rate loads its Test-best checkpoint.
The exact availability stored in the source prediction artifact is replayed;
one unchanged causal Memory scan supplies the Flat Adapter input. Only Base
and/or all Gap slots are zeroed before the trained Adapter.

- `LBG`: Adapter Local + Base + Gap (Full)
- `LG`: Adapter Local + Gap; Base removed
- `LB`: Adapter Local + Base; Gap removed
- `L`: Adapter Local only; Base and Gap removed

All eight Full replays reproduce their saved predictions exactly, with zero
class disagreement. The replay mask is the final saved stochastic realization,
not the unavailable mask realization that originally established each
Test-best maximum; therefore the analysis relies on paired deltas.

## Accuracy (%)

| Condition | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Mean | High |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LBG | 66.284 | 64.253 | 60.766 | 60.192 | 59.923 | 56.513 | 55.517 | 54.444 | **59.737** | **55.492** |
| LG (no Base) | 66.015 | 63.678 | 60.996 | 60.115 | 59.655 | 55.939 | 54.866 | 54.330 | 59.449 | 55.045 |
| LB (no Gap) | 66.284 | 64.061 | 60.996 | 60.230 | 59.540 | 56.475 | 54.943 | 54.406 | 59.617 | 55.275 |
| L (no Base/Gap) | 66.015 | 63.563 | 61.111 | 59.693 | 59.540 | 55.785 | 54.789 | 54.330 | 59.353 | 54.968 |

## Weighted F1 (%)

| Condition | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Mean | High |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LBG | 64.172 | 61.111 | 58.595 | 55.780 | 56.440 | 50.129 | 49.915 | 46.489 | **55.329** | **48.845** |
| LG (no Base) | 63.932 | 60.313 | 58.709 | 55.672 | 55.541 | 49.127 | 48.215 | 46.183 | 54.711 | 47.842 |
| LB (no Gap) | 64.172 | 60.923 | 58.707 | 55.780 | 55.876 | 49.895 | 48.877 | 46.291 | 55.065 | 48.354 |
| L (no Base/Gap) | 63.932 | 60.165 | 58.802 | 55.119 | 55.418 | 48.864 | 48.063 | 46.190 | 54.569 | 47.706 |

## Paired contribution summary

| Intervention | Mean ACC delta | Mean W-F1 delta | High ACC delta | High W-F1 delta | Corrections | Harms | Changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| Remove Base | -0.288 | -0.618 | -0.447 | -1.003 | 308 | 368 | 990 |
| Remove Gap | -0.120 | -0.264 | -0.217 | -0.491 | 96 | 121 | 334 |
| Remove Base and Gap | -0.384 | -0.760 | -0.524 | -1.139 | 353 | 433 | 1,173 |

Counts pool the eight rate evaluations. A correction means Full was wrong and
the intervention became correct; a harm means Full was correct and the
intervention became wrong. Removing evidence produces both, but harms exceed
corrections for every aggregate intervention.

## Conclusion

After retraining without the final Local shortcut, the remaining Adapter does
use both historical evidence groups. Base has the larger paired contribution:
removing it costs 0.618 W-F1 points overall and 1.003 points at high missing,
whereas removing Gap costs 0.264 and 0.491 points. Removing both costs 0.760
points overall. These effects remain modest, and Adapter Local is still present
in every condition; this experiment therefore does not show that Memory alone
is sufficient.

Remote artifact:

```text
/data2/yb/paper/remote_runs_20261012/results/
osram_meld_gcnet_random_no_skip_evidence_20261014/seed_66_corrected
```
