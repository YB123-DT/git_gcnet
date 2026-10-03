# Meaningful twenty-block continuous search

User update (2026-10-03): **at most 60 distinct meaningful methods, twenty per
round, at most three rounds**. This supersedes all earlier unlimited-loop
wording. Replication seeds/retries are not new methods. At the cap, finish the
authorized replication/results and stop; no automatic fourth round. Use one
shared minimal correctness/real-batch preflight template; passing candidates
may start immediately without waiting for other candidates or new review rounds.

Plan approved sequentially by Architect and independent Critic, 2026-10-03. Implementation/testing is in progress; approval is not training/performance verification.

- [中文执行摘要](PLAN.zh.md)
- [English execution summary](PLAN.en.md)
- [Canonical PRD](../../.omx/plans/prd-osram-meaningful20-loop.md)
- [Canonical test specification](../../.omx/plans/test-spec-osram-meaningful20-loop.md)

The user has authorized continuous source → implementation → fixed-protocol experiments → replication → next-round work. Sequential Architect then Critic review passed. Existing basic-operator screen stays withdrawn. Twenty source/design cards are accepted across six families; this does not claim training results.

After review, freeze twenty distinct accepted non-MSA/MERC designs before the first training run of each round. Individual candidates may start after their own implementation/CPU/CUDA checks while other implementations continue. Finish all twenty seed-66 runs before fixed primary-metric ranking and at most three promotions. If no three-seed mean improvement survives, begin another distinct twenty without requesting permission again.
