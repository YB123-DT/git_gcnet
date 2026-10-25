# Completed MOSI seed66 matched retraining

Corrected execution_v2 completed8/8formal150epoch branches at
2026-10-25T16:10:33Z,about4minutes14seconds including real smoke/staggered startup.
Only frozen-parent branch parameters train; this is not an OSRAM full retrain.
No added seeds,baseline retraining,HPO or changed Missing protocol.

| Scope | Model | ACC | W-F1 |
| --- | --- | ---: | ---: |
| Eight-rate mean | Flat | 79.535061 | 79.460447 |
| Eight-rate mean | Full Signature | 79.858994 | 79.847119 |
| Eight-rate mean | Retrained NoArea | 79.935213 | 79.919133 |
| High0.5–0.7 | Flat | 73.475610 | 73.351571 |
| High0.5–0.7 | Full Signature | 73.932927 | 73.960737 |
| High0.5–0.7 | Retrained NoArea | 74.186992 | 74.227697 |

NoArea-full differences: +0.076220ACC/+0.072014WF1 points overall and
+0.254065ACC/+0.266960WF1 high-missing. NoArea-Flat: +0.400152ACC/+0.458687WF1
overall. Individual rates have mixed signs (0.0/0.4 lower than full,0.2/0.5/0.7
higher,three ties),so this is not uniform rate-wise dominance or statistical
superiority. The seed66 result agrees with seed68 in not supporting directional
antisymmetric area as necessary for the observed aggregate branch benefit.
It still does not separate endpoint,time,count,symmetric quadratic features,
Local conditioning or extra learned capacity. No one-stage/generalization claim.

VERIFICATION.json independently recomputes ACC/WF1 from saved predictions and
confirms all150epoch masks/sample/update/rate/LR traces match original full
Signature. Selected weight/prediction hashes match;labels,availability,metadata
and frozen Flat outputs are array-identical between paired conditions. All8jobs
complete with finite-gradient/replay/frozen-parent checks. Local and remote18
focused tests and real two-update CUDA smoke passed on corrected source6314804.

First source68ab183/execution_v1 failed only its runtime shared-module import,
before any formal training or data forward. That failed source/log is preserved,
documented in README,and excluded from results; corrected source/execution_v2
uses the same original mathematical/training protocol. No existing artifacts
were overwritten,no other users' processes signalled. GPU0/1/2 only.

Remote weights,full Adam/RNG last states,exact masks and predictions remain under
the LAUNCH.json output path. Only code,tests and small JSON/CSV/text evidence are
committed. Test-best results remain INTERNAL per-rate Test-oracle exploration;
eight-rate means combine separate branch/parent pairs,not one checkpoint.
