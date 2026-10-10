# Original Nested: 100 to 150 epochs

INTERNAL DIAGNOSTIC ONLY

Status: all three seeds completed150epochs, exited and final provenance verifies
all20required artifacts. Full model/optimizer/RNG/schedule/selection state resumed
from epoch100 in independently copied directories. Historical source ad211c0;
continuation wrapper f6e4f68. Only epoch budget changed, with constant LR unchanged.
No Gate, new loss, model modification or Flat training. Original100epoch files
remain untouched. Server biggpu, physical GPU7.

## W-F1 (%)

| Seed | 100epoch 8-rate | 150epoch 8-rate | Delta pp | 100epoch high | 150epoch high |
|---|---:|---:|---:|---:|---:|
| 66 | 80.992147 | 81.001067 | +0.008920 | 76.077229 | 76.077229 |
| 67 | 80.850540 | 80.850540 | 0 | 75.497246 | 75.497246 |
| 68 | 79.623766 | 79.623766 | 0 | 74.180973 | 74.180973 |
| Mean | 80.488818 | 80.491791 | +0.002973 | 75.251816 | 75.251816 |

High: rates .5/.6/.7. Metrics use eight independently selected per-rate
Test-oracle BEST checkpoints from each seed's one cyclic training run.
BEST considers all150epochs, including the original100. Expanding this selection
window cannot lower retained BEST; tiny improvements alone do not prove better
generalization or a better architecture.

Only seed66 miss=.3 improved, from80.522617 to80.593977 (+0.071360pp), with BEST
moving from epoch94 to127. The other23seed/rate BEST epochs remain unchanged.
The high-missing means are identical. Extra50epochs give no practically meaningful
aggregate gain in these three runs; this does not prove all longer runs are futile.

## Selected BEST epochs

| Miss | Seed66 100→150 | Seed67 100→150 | Seed68 100→150 |
|---|---|---|---|
| .0 | 91→91 | 55→55 | 73→73 |
| .1 | 84→84 | 47→47 | 73→73 |
| .2 | 84→84 | 47→47 | 92→92 |
| .3 | 94→127 | 47→47 | 83→83 |
| .4 | 71→71 | 47→47 | 94→94 |
| .5 | 62→62 | 47→47 | 76→76 |
| .6 | 71→71 | 48→48 | 73→73 |
| .7 | 98→98 | 49→49 | 88→88 |

Per-rate scores are saved in SUMMARY.json and original/final METRICS files.
Completion checks enforce150history entries, unchanged evaluation mask hashes,
retained100epoch BEST, all expected checkpoints/predictions, and unchanged
original last_training checkpoint hashes. Summarization adds no training/inference.

Recompute: `python experiments/osram_nested_extend150_20261010/summarize.py`.
No additional extension or parameter search is launched.
