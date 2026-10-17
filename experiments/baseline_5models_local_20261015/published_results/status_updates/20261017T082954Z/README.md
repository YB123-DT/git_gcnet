# Live execution evidence 20261017T082954Z

Formal status at this timestamp: {"PENDING": 282, "COMPLETED": 6, "FAILED": 2, "RUNNING": 4}. All26 GPU smokes passed. The original294-job supplement queue is active on local GPUs0/1.

Two MELD MMIN MR0.6 runs failed in the copied original GCNet mask function on the50-utterance final test batch. The same real checkpoint RNG produces the same division-by-zero OverflowError and post-failure RNG in official GCNet. No guard, retry, seed change, batch-size change or sampler substitution was added. Failed runs remain failed; other jobs continue. See the included audit and regression test.

The official AIM3 MMIN data recipe enumerates six missing patterns; random-MR dynamic sampling here is the explicitly requested shared experimental protocol adapter, not a claim that its official data recipe was unchanged. Model architecture files remain unchanged.

Current completed MELD scores and incomplete seed groups are in MELD_RESULTS.md and CSVs. All metrics are from the same selected epoch. Test-oracle bias remains.
