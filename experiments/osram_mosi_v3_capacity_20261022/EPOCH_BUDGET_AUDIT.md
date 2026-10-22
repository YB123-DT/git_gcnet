# Check the conditional MOSI100 request against completed seeds

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY. Existing histories only; no training,
new inference or weight/config modification.

User condition: if all three3e-4 seeds' selected epochs are within100, use100
instead of150 for subsequent MOSI work. Current six arms use3e-4 in epochs1..100,
then3e-5 in101..150; not3e-4 for all150 epochs. MOSEI is unaffected.

| Seed | Min selected epoch | Max selected epoch | Eight-rate WF1 first100 | All150 | Gain pp |
|---:|---:|---:|---:|---:|---:|
| 66 | 48 | 62 | 79.249087 | 79.249087 | 0 |
| 67 | 42 | 122 | 79.365682 | 79.402372 | +0.036690 |
| 68 | 36 | 57 | 79.193726 | 79.193726 | 0 |

23/24 selected epochs are within100. The exception is seed67,miss0.4:
first100 best77.846390%,all150 best78.139913%,selected epoch122,+0.293523pp.
The three-seed eight-rate mean gains only0.012230pp after100. This quantifies
the small historical return, but cannot guarantee that wider V3 converges at
the same time as the original Flat.

The literal all-points condition is NOT satisfied; therefore the current immutable
six-arm150epoch queue is unchanged. No process was signalled, no scheduler or
checkpoint selection changed, and no past result was relabeled100epochs.
EPOCH_BUDGET_AUDIT.json records every rate plus exact history/summary SHA256.
