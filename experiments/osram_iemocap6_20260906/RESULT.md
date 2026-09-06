# OSRAM IEMOCAP-6 result

**Internal diagnostic only; not a formal paper result.**

The table reports five-seed mean ± sample standard deviation of weighted F1
(percentage points). Each rate uses its own Test-oracle epoch extracted from
the per-epoch histories; it is not one common checkpoint across rates.

| Missing rate | W-F1 mean ± SD (%) |
|---:|---:|
| 0.0 | 67.11 ± 0.91 |
| 0.1 | 66.43 ± 1.15 |
| 0.2 | 65.55 ± 0.70 |
| 0.3 | 65.09 ± 1.02 |
| 0.4 | 65.11 ± 0.69 |
| 0.5 | 63.84 ± 1.19 |
| 0.6 | 63.14 ± 1.17 |
| 0.7 | 62.78 ± 0.37 |

The eight-rate mean is **64.88%**. The high-missing mean over 0.5/0.6/0.7 is
**63.25%**. Per-seed values and rate-specific selected epochs are in
`per_seed_rate.csv`; the raw histories are under `raw/`.

Because IEMOCAP `official` uses the held-out Session 5 for both validation and
test, and because this table independently selects each rate on Test, these
values are only a diagnostic of whether OSRAM is worth further work.
