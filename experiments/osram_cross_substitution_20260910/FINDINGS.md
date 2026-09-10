# Base/Gap direct substitution: partial, asymmetric functional replaceability

Completed evaluation-only on five existing MOSI causal Flat eta=.6 checkpoints.
All eight rates were evaluated; rate 0 has no one-missing samples and is excluded
from the following descriptive mean. Average rates .1–.7 within each seed, then
average seeds 66–70. This is an **AT/AV/TV subset statistic**, not a whole-test score.

| Intervention | W-F1 (%) | Relative to normal (pp) |
|---|---:|---:|
| Normal | 79.182 | — |
| Remove Gap | 78.352 | -0.829 |
| Remove Base | 78.119 | -1.062 |
| Copy Base into active Gap slot, retain Base | 79.035 | -0.147 |
| Copy active Gap into Base slot, retain Gap | 78.642 | -0.540 |

Base→Gap recovers 0.683 pp versus removing Gap (5/5 seeds positive).
Gap→Base recovers 0.523 pp versus removing Base (4/5 seeds positive).
Thus the frozen classifier can use a substantial portion of Base in the Gap slot;
the reverse substitution is less complete. This supports **partial asymmetric
functional replaceability**, not proof of equal geometry or semantic redundancy.
Recovery is not uniform across rates: Base→Gap is worse than removing Gap at .4.
Neither substitution is consistently superior to the normal two-role readout.

No new training, gates, loss, memory updates or representation-geometry statistics.
Only `emotion_adapter` input is patched in an offline replay of the saved contexts;
all cases share the same original scan. No patched tensor is passed back to OSRAM.
Normal logits match original forward bit-for-bit in all evaluated batches.
All 40 seed/rate test masks match the existing reference hashes.

## Important checkpoint qualification

Only `best.pt` is available in each reference run. Those checkpoints were selected
by eight-rate-mean Test W-F1 historically. They are fixed across all interventions
and rates here. No intervention-specific or per-rate epoch is selected; no missing
checkpoint is reconstructed. Do not compare this subset mean to the earlier
per-rate Test-oracle 80.002 whole-test mean.

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**

## Artifacts and verification

- `run.py`: evaluation-only collection and slot patching; production model untouched.
- `summarize.py`: reproducible per-seed/rate tables and recovery contrasts.
- `results/RESULT.md`: full tables, checkpoint epochs and qualifications.
- `results/seed*.json`: checkpoint SHA256, exact-replay/mask checks, pattern metrics.
- `results/seed*_rate*.npz`: five outputs on identical samples, labels and availability.
- `results/per_seed_rate.csv`, `results/summary.csv`: raw/grouped W-F1, counts and SD.
- `tests/test_osram_cross_substitution.py`: 1 targeted red-to-green test passed;
  checks five modes, donor retention, active slot, ineligible/padding rows and no mutation.

Execution used the existing remote s0 Python and GPU 2. No dependency changes.

Reproduce from the repo root with a **new output directory**:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. /data2/yb/reproduction_envs/s0/bin/python3.10 \
  experiments/osram_cross_substitution_20260910/run.py --output /path/to/new-output
python3 experiments/osram_cross_substitution_20260910/summarize.py /path/to/new-output
```

Remaining limitations: only one-missing patterns, one dataset, fixed historically
Test-selected checkpoints; substitution can be influenced by slot decoding and
input distribution. No significance or causal claim about semantic identity.
