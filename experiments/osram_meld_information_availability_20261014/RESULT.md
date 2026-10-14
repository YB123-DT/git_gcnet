# MELD Information Availability Probe

**INTERNAL DIAGNOSTIC ONLY.** The retained backbone checkpoints and offline
probe epochs are Test-oracle selected. These numbers diagnose the fixed
representations; they are not deployable model scores or independent paper
results.

## Protocol and verification

- Dataset/backbone: MELD cfg84 no-JEPA causal Flat, backbone seed 66.
- Missing rates: 0.0--0.7 with the exact retained `gcnet-legacy-fixed-v1`
  masks and one checkpoint per rate.
- Frozen inputs: current Local (256), forward Base (512), and hard-masked
  forward Gap-A/T/V (3×512). The fixed backward-zero half is excluded.
- Raw-history control: strict-past mean Local over all speakers and over the
  current speaker, with history-presence/count metadata.
- Donor control: same split, different conversation, matched current
  availability and history-presence state; no labels used for matching.
- Probe: `LayerNorm -> Linear(64) -> GELU -> Linear(7)`, 30 epochs, train-only
  normalization, probe seeds 66/67/68.
- Completed: 8 rates × 13 variants × 3 probe seeds = 312 fits. Only 6/312
  selected epoch 30, so the fixed budget did not generally truncate optima.
- Source parity: exact ACC and W-F1 parity passed at all eight rates; the frozen
  model hash was unchanged after every run. Donor coverage was at least
  2608/2610 test utterances and was identical on each paired comparison.
- Code: training/extraction commit `68c698f`; report renderer commit `3cbf6d7`.
- Full artifacts: `/data2/yb/remote_experiments/osram_meld_information_availability_20261014/full_68c698f_v2`
  (approximately 709 MB, including caches, predictions and probe weights).

## Eight-rate and high-missing results

Mean ± sample standard deviation across three probe initializations, in percent.

| Probe | ACC mean±sd | W-F1 mean±sd | high ACC | high W-F1 |
|---|---:|---:|---:|---:|
| Base | 50.981±0.137 | 43.146±0.430 | 48.489 | 37.152 |
| BaseDonor | 48.151±0.010 | 31.401±0.045 | 48.144 | 31.378 |
| Local | 60.102±0.056 | 56.597±0.104 | 55.492 | 49.839 |
| LocalBase | 59.765±0.058 | 56.620±0.130 | 55.028 | 50.052 |
| LocalBaseDonor | 59.521±0.028 | 56.013±0.205 | 54.947 | 49.398 |
| LocalMemory | 59.516±0.098 | 56.257±0.087 | 55.036 | 49.810 |
| LocalMemoryDonor | 59.208±0.228 | 55.374±0.138 | 54.721 | 48.803 |
| LocalRawHistory | **60.330±0.099** | **57.068±0.139** | **55.981** | **50.926** |
| LocalRawHistoryDonor | 59.909±0.050 | 56.148±0.040 | 55.287 | 49.489 |
| Memory | 53.292±0.106 | 47.120±0.169 | 51.916 | 44.186 |
| MemoryDonor | 48.012±0.044 | 31.438±0.086 | 47.961 | 31.467 |
| RawHistory | 48.333±0.089 | 35.555±0.305 | 48.450 | 34.946 |
| RawHistoryDonor | 48.150±0.007 | 31.358±0.032 | 48.161 | 31.383 |

## Per-rate ACC

| Probe | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local | 65.990 | 64.227 | 63.844 | 60.881 | 59.400 | 56.922 | 55.964 | 53.589 |
| LocalBase | 65.530 | 64.021 | 63.793 | 61.188 | 58.506 | 56.501 | 55.492 | 53.091 |
| LocalMemory | 65.479 | 63.369 | 63.538 | 60.447 | 58.186 | 56.590 | 55.479 | 53.040 |
| LocalRawHistory | **66.245** | **64.379** | 63.729 | **61.175** | 59.170 | **57.395** | **56.526** | **54.023** |
| Base | 59.132 | 54.537 | 50.396 | 48.927 | 49.387 | 48.672 | 48.467 | 48.327 |
| Memory | 59.119 | 54.627 | 51.890 | 52.669 | 52.286 | 51.648 | 52.363 | 51.737 |
| RawHistory | 47.931 | 47.942 | 48.761 | 48.404 | 48.276 | 48.442 | 48.429 | 48.480 |

The full donor per-rate ACC table is retained in remote `SUMMARY.json`; the
paired deltas below are the relevant capacity-matched comparison.

## Per-rate W-F1

| Probe | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local | 64.811 | 62.685 | **61.605** | 57.980 | **56.175** | 52.417 | 50.238 | 46.863 |
| LocalBase | 64.429 | 62.710 | 61.452 | **58.538** | 55.677 | 52.293 | 50.591 | 47.273 |
| LocalMemory | 64.617 | 61.949 | 60.962 | 57.688 | 55.409 | 51.780 | 50.487 | 47.163 |
| LocalRawHistory | **65.133** | **63.058** | 61.375 | 58.469 | 55.729 | **52.701** | **51.570** | **48.506** |
| Base | 55.862 | 50.757 | 44.549 | 42.018 | 40.524 | 37.719 | 37.440 | 36.296 |
| Memory | 56.230 | 50.310 | 45.843 | 46.203 | 45.811 | 43.891 | 45.470 | 43.198 |
| RawHistory | 37.438 | 36.893 | 36.189 | 34.252 | 34.834 | 35.394 | 34.773 | 34.670 |

## Paired real-minus-donor deltas

| Real | Control | ΔACC | ΔW-F1 | min N |
|---|---|---:|---:|---:|
| Base | BaseDonor | +2.829 | +11.744 | 2608 |
| Memory | MemoryDonor | +5.280 | +15.681 | 2608 |
| LocalBase | LocalBaseDonor | +0.244 | +0.607 | 2608 |
| LocalMemory | LocalMemoryDonor | +0.308 | +0.883 | 2608 |
| RawHistory | RawHistoryDonor | +0.184 | +4.198 | 2608 |
| LocalRawHistory | LocalRawHistoryDonor | +0.421 | +0.920 | 2608 |

| Real | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base ΔW-F1 | +24.284 | +19.454 | +13.145 | +10.575 | +9.177 | +6.326 | +6.113 | +4.880 |
| Memory ΔW-F1 | +24.740 | +18.783 | +14.461 | +14.848 | +14.461 | +12.316 | +14.013 | +11.828 |
| LocalBase ΔW-F1 | +0.003 | +0.352 | +0.911 | +1.595 | +0.037 | +0.571 | +1.114 | +0.277 |
| LocalMemory ΔW-F1 | +0.242 | +0.564 | +1.125 | +1.627 | +0.484 | +0.880 | +1.426 | +0.715 |
| RawHistory ΔW-F1 | +6.170 | +5.521 | +4.861 | +2.890 | +3.451 | +3.986 | +3.446 | +3.255 |
| LocalRawHistory ΔW-F1 | +0.326 | +0.621 | +0.477 | +1.228 | +0.399 | +1.218 | +1.540 | +1.551 |

## Diagnostic conclusion

The pre-registered rule yields **Case C**:

- strict-past RawHistory beats its donor at every rate and for every probe seed,
  so Case A (no readily decodable historical signal) is rejected;
- learned Base/Memory beat their donors substantially, so the result does not
  support a simple storage-failure account (Case B);
- `Local+Memory` beats the exact-width `Local+MemoryDonor` control by
  **+0.308 ACC / +0.883 W-F1**, consistently across probe seeds and rates,
  while P0 found little realized gain through the original Flat path.

This is nevertheless **weak/qualified Case-C evidence**, not proof that a new
readout will improve the task. `Local+Memory` (56.257 W-F1) remains slightly
below the simpler `Local` probe (56.597), whereas `Local+RawHistory` reaches
57.068. Thus real Memory carries sample-specific conditional information, but
its incremental value is modest and does not overcome the optimization/capacity
cost of the wider probe. The cleanest next research question is why a simple
strict-past history summary complements Local better than the learned OSRAM
readout—not another generic Gate or larger fusion block.
