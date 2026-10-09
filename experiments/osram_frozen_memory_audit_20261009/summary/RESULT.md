# TEST-ORACLE INTERNAL DIAGNOSTIC

Partial: False. Completed rates: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]. Freeze and baseline parity verified: True.

Three probe seeds, not three independently trained backbones. Test-selected probe epochs are internal diagnostics, not independent generalization.

Frozen original backbone seed 66. Probe gradients use train; best probe epochs use minimum Test MSE. No validation selection.

Probe heads are approximately capacity matched: A 33,409 parameters; B/C 33,089. The fixed random 512→64 per-slot projection limits what probes can recover; null gains do not establish absent memory information.

Sign metrics for historical and difference score targets are not current sentiment classification or evidence of relational reasoning.

Metrics use raw score MSE and nonzero-target W-F1/ACC (fractions). Flips, corrections and harms exclude zero targets. A/C→B compare matched conversation/utterance identities and original rows.

mean8 = rates 0.0–0.7; high = 0.5–0.7. Available rates receive equal weight; coverage and seed SD are reported explicitly. Incomplete aggregates are descriptive only.

Intervention intervals use 500 conversation-cluster bootstrap resamples separately per rate. Exact-lag modality comparisons are separate from all eligible pairs. Coverage and skipped reasons are preserved in SUMMARY.json.

## Current-score probe results

| Scope | Probe | MSE mean ± SD | W-F1 mean ± SD | ACC mean ± SD | Seeds |
|---|---|---|---|---|---|
| high | A | 1.777151 ± 0.007578 | 0.725106 ± 0.002868 | 0.724255 ± 0.003105 | 3 |
| high | B | 1.664378 ± 0.003332 | 0.745721 ± 0.002629 | 0.744919 ± 0.002829 | 3 |
| high | C | 1.824953 ± 0.012936 | 0.722391 ± 0.012664 | 0.721545 ± 0.013064 | 3 |
| mean8 | A | 1.453662 ± 0.001611 | 0.780823 ± 0.001505 | 0.780615 ± 0.001405 | 3 |
| mean8 | B | 1.371234 ± 0.000530 | 0.797855 ± 0.001175 | 0.797764 ± 0.001352 | 3 |
| mean8 | C | 1.493912 ± 0.002359 | 0.778519 ± 0.004295 | 0.778201 ± 0.004537 | 3 |

Per-rate scores, all four score targets, matched correction/harm counts, intervention families/patterns and integrity checks are in the CSV tables and SUMMARY.json. These diagnostics alone do not establish performance improvement or memory reasoning.
