# Live finite experiment execution

INTERNAL TEST-ORACLE SCREEN. This file records execution, not performance.

Inspected branch HEAD e0bb6b4. Initial source f24c86c failed pre-training smoke;
zero formal jobs used that version. Regression-verified correction7abb855 is
the ONLY formal version. Independent corrected output:
`/data2/yb/remote_experiments/osram_meld_block2_auto_20261014_v2`.
Immutable code:
`/data2/yb/remote_experiments/osram_meld_block2_code_7abb855`.

77remote CPU contracts pass, including original GCNet bit/RNG equivalence,
exact epoch artifacts, adapters/history/masks/delayed gradients, recovery,
queue/PID/resource limits, and actual Optuna SQLite protocol integration.
GPU smoke measures32longest real train conversations, 33padding steps,
three actual task optimizer updates. All seven succeed. Peak MiB:

| Flat | B1 | B2 | B3 | B4 | B5 | B6 |
|---:|---:|---:|---:|---:|---:|---:|
| 607.573 | 1497.426 | 779.012 | 1190.967 | 702.675 | 749.752 | 947.751 |

Formal controller PID1203491 (nohup, SSH-independent). Seven jobs launched:

| Method | Host GPU | PID |
|---|---:|---:|
| Flat | 0 | 1211648 |
| B1 | 2 | 1211652 |
| B2 | 3 | 1211660 |
| B3 | 0 | 1211676 |
| B4 | 2 | 1211680 |
| B5 | 3 | 1211684 |
| B6 | 0 | 1211690 |

UUIDs and effective configs/source/data hashes are in evidence/PREFLIGHT.json
and GPU smoke records. Running controller/queue live at RUNNER.json and
queue/QUEUE.json remotely; evidence/RUNNER.json is a point-in-time capture,
not a claim that the job is forever running. All seven reached real training
and wrote initial epoch histories without failure. Final scores/HPO are
pending as of this execution-record commit.

Controller log: `<output>/controller.log`; per-job logs:
`<output>/queue/logs/screen_<method>.attempt1.log`.
It will complete fiftyepochs, paired on/off replay, fixed Top2 selection and
conditional twenty-trial training HPO. No external task was stopped. Original
baseline24checkpoints remain outside all cleanup scopes.
