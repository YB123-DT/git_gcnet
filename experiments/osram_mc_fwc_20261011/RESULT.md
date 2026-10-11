# MC-FWC — pending formal results

Implementation and verification completed; formal training is running, not finished.
No performance gain is established by this report.

- Code: `e9a7949`, based on `c838e14`.
- 69 targeted tests passed; five-mode real MOSI GPU smoke passed.
- E0–E4 × seeds66/67/68, 150 epochs each; one validation-selected checkpoint/run.
- Four concurrent jobs on biggpu GPUs1/2; remaining jobs in a finite automatic queue.
- At the initial progress check, E0/E4/E2/E3 seed66 had completed 6/4/3/3 epochs.
- No test performance has been read or used to alter the configuration.

See [REPORT.md](REPORT.md) for methods, controls, test evidence, commands and risks;
[LAUNCH.md](LAUNCH.md) for remote paths and recovery instructions. Completed
per-rate, high-missing, persistent, transition and utility results will reside in
the remote suite's SUMMARY.json, RESULT.md and per-run metrics/prediction files.
Do not confuse the engineering smoke or early validation scores with final results.
