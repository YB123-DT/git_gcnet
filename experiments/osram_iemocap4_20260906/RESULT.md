# OSRAM IEMOCAP-4 result

**Internal diagnostic only; not a formal paper result.**

The table reports five-seed mean ± sample standard deviation of weighted F1
(percentage points). Each rate uses its own Test-oracle epoch extracted from
the per-epoch histories; it is not one common checkpoint across rates.

| Missing rate | W-F1 mean ± SD (%) |
|---:|---:|
| 0.0 | 86.26 ± 0.46 |
| 0.1 | 86.14 ± 0.56 |
| 0.2 | 85.47 ± 0.77 |
| 0.3 | 84.76 ± 0.33 |
| 0.4 | 84.18 ± 0.61 |
| 0.5 | 82.56 ± 1.70 |
| 0.6 | 82.61 ± 0.95 |
| 0.7 | 82.13 ± 0.63 |

The eight-rate mean is **84.26%**. The high-missing mean over 0.5/0.6/0.7 is
**82.43%**. Per-seed values and rate-specific selected epochs are in
`per_seed_rate.csv`; the raw histories are under `raw/`.

Because IEMOCAP `official` uses the held-out Session 5 for both validation and
test, and because this table independently selects each rate on Test, these
values are only a diagnostic of whether OSRAM is worth further work.
