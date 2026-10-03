# Independent infrastructure review

Status: **REQUEST CHANGES / rerun pending**, 2026-10-03. Three material findings
were reproduced or traced and sent to the infrastructure author, who is
implementing focused fixes. This record is not training authorization.

Scope: `run.py`, `queue.py`, `preflight.py`, `manifest.py`,
`gcnet_missing_m3/training_resume.py`, relevant CPU tests and trainer recovery
call sites. Reviewer changed only this review record. No GPU/SSH/Git operation.

## Material findings at reviewed revision

1. **Baseline/data audit identity was not carried across the launch boundary.**
   `queue.py:125-135` computed and stored baseline-audit/data-manifest hashes,
   but the loop revalidated only the method manifest. The child command at
   `queue.py:198-204` supplied paths but not expected audit/data hashes, and
   `run.train` reread those files. Changing either JSON between coordinator
   startup and child launch could select different data/references while the
   coordinator continued using old baseline means. Required fix: recheck all
   pinned spec files on each dispatch cycle and pass/check their expected
   hashes in the child before consuming their contents.
2. **Crash-window disk reservation was incomplete.** The pre-Popen
   `launch_intent` at `queue.py:207-212` omitted the already validated profile;
   only the post-Popen update at line222 added it. A coordinator crash in this
   window could recover a live job from child identity but reserve zero future
   checkpoint space. Required fix: profile must be part of the durable intent;
   legacy/invalid active jobs without a usable reservation must fail closed.
3. **Observed abnormal exit did not override complete artifacts.** The queue
   stored `process_exit_code` at lines138-140 but `reconcile_job:48-49` ignored
   it. A child failing after writing its completion record could be promoted.
   Required fix: nonzero observed exit rejects completion; unknown exit after
   coordinator restart must not be silently equated with normal exit.

These line numbers identify the pre-fix reviewed code and may shift when the
author lands changes. The essential regression scenarios are semantic.

## Reproducible evidence

The following CPU-only probe printed `complete` for the first case and
`running, 0.0` for the second before fixes:

```python
from unittest.mock import patch
import os
from experiments.osram_meaningful20_20261003 import queue as q
job = dict(pid=-99, status='running', output='/missing_review_fixture',
           process_exit_code=1)
with patch.object(q, 'completion_status', return_value=('complete', None)):
    print(q.reconcile_job(job)['status'])
intent = dict(status='launch_intent', output='/missing_review_fixture',
              run_id='fixture', gpu_uuid='healthy',
              **q.process_identity(os.getpid()))
recovered = q.reconcile_job(intent)
print(recovered['status'], q.reserved_disk_gib([recovered]))
```

## Verified protections and non-blocking observations

- Snapshot execution binds `run.py`'s actual repository path to the verified
  snapshot root; recorded file hashes and unrecorded Python files are checked.
  Source symlinks/path escapes are rejected. Dirty unrelated source can be
  explicitly replaced by committed bytes without editing the worktree.
- Queue/output locks, PID/start-ticks/boot-ID checks, atomic launch intent,
  child identity recovery and inspection-pending ambiguous launches avoid
  blind duplicate dispatch. Fixed healthy indices and the permanent bad-card
  UUID are now both checked (the author's concurrent UUID fix was rerun).
- The author added explicit dataset-root binding before trainer imports,
  canonical label/split hashing and inherited-config mismatch rejection.
  This focused change passes the data-binding regression; it does not by
  itself resolve the launch-boundary manifest-hash finding above.
- Memory margins, compute/temperature checks and disk reserve are explicit;
  active-run future disk subtraction is appropriate once intent profiles are
  durable. No automatic artifact deletion or protocol shrink is present.
- `TrainingState` uses immutable model/selection versions, atomic last-state
  commit and canonical BEST/history repair. Identity, required state,
  scheduler/scaler applicability and reference hashes fail closed. The
  actual trainer binds sampler/mask identities and restores RNG before its
  next epoch. CPU recovery evidence below supports this contract.
- The next-round state intentionally returns control to the authorized leader
  for new research rather than fabricating twenty new cards in a script.
- Human source/novelty acceptance is still required: JSON validation checks
  count/IDs/hashes/evidence presence, not whether two differently described
  papers are substantively the same method. This is an expected review gate,
  not a semantic guarantee supplied by `validate_round`.

## Commands executed

Using `/home/yangbin/miniconda3/envs/multimodalerc310/bin/python`:

```text
-m unittest discover -s tests -p test_meaningful20_infrastructure.py
  Initial run: 17 tests, two known in-progress UUID/data-binding failures.
  After those author fixes: 17 tests in0.064s — OK.
-m unittest discover -s tests -p test_meaningful_resume.py
  5 tests in1.047s — OK.
-m unittest discover -s tests -p test_meaningful20_runner.py
  2 tests in1.637s — OK (real tiny CPU trainer, uninterrupted versus resumed).
```

Outstanding: independent rerun of the three material-finding regressions after
the author lands them. GPU health/UUID runtime probes, real-shape CUDA memory,
remote filesystem capacity and actual long-run recovery were not performed.
