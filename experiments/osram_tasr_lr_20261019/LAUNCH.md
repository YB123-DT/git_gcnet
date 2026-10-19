# Launch evidence

2026-10-19, biggpu. Training source `be93ece426cfb9848b0b22b06bb0bbd757154d28`
was committed and pushed to `github` (`YB123-DT/git_gcnet`), branch
`feature/ecerc-features`. Original `origin` upstream was not pushed.

Immutable snapshot:
`/home/qinjinghui/remote_experiments/osram_tasr_lr_20261019/source_be93ece`

Persistent tmux session: `tasr_lr_be93ece`.
Output: `/home/qinjinghui/remote_experiments/osram_tasr_lr_20261019/runs_v1`.
Launcher log: sibling `launcher.log`. Scheduler state: `runs_v1/queue/QUEUE.json`.

Confirmed queue running with four UUID-bound GPU1 workers and two pending,
not six simultaneously. Scheduler automatically admits remaining jobs when a
slot is free. Physical GPU1 UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`.
Initial observed GPU1 usage2718MiB/util98%; this is a startup observation, not
the peak. GPU4 prohibited; other users' jobs untouched.

| Job | Initial PID | Initial status |
|---|---:|---|
| MELD LR3e-6 |46897|running|
| MELD LR1e-5 |46900|running|
| MELD LR3e-5 |46903|running|
| IEMOCAP6 fold5 LR3e-6 |46906|running|
| IEMOCAP6 fold5 LR1e-5 |—|pending automatic admission|
| IEMOCAP6 fold5 LR3e-5 |—|pending automatic admission|

Historical PIDs are not authority for future signaling. Consult queue birth
identity and current artifacts; never kill a reused/external PID.

Preflight: GPU1 initially32491MiB free; home filesystem50GiB free; data
filesystem3.8GiB free, so all new outputs use home. Runtime remains
`/data2/yb/conda_envs/cmerc-repro/bin/python`, Torch1.13.1+cu117/NumPy1.26.4.
Both original completed F2 SUMMARY files present, not replaced.

Local focused TASR suite:56passed with one existing optional torch-sparse
warning (`libcusparse.so.11` unavailable). New runner/queue/protocol tests on
the remote actual runtime:25passed. Compileall and git diff --check passed.
No new inference module or structural GPU smoke introduced; old verified
OSRAM/TASR core hashes must match and each real job checks anchor/zero-init
parity, finite Adapter gradients and unchanged frozen backbone.

Remote/local source file SHA256 equality verified:

- runner: `9581db93f49cc9c73cb9a1cc3ebb36c4fb5f8e5a39b7aea46178643e6fc8949b`
- LR queue: `58eb031f9ab81186cc6038d1a7a968d1e01080f046a4e876c547d753800d2467`

These are launched training jobs, not completed results. Queue completion
automatically generates LR_RESULTS.json/CSV and RESULT.md. Check best_adapter.pt,
last.pt, history.json, SUMMARY.json and evaluation outputs for real completion.
Do not invent new scores or call this a performance improvement before results.

This follow-up changes F2 LR only. F1 has not received matched LR tuning;
comparison against old F1 cannot isolate source decomposition from optimization.
Validation selection and all limitations from the original TASR protocol remain.
