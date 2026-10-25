# Verified finite launch

Launched on biggpu at2026-10-25T15:49:08Z in persistent tmux
`signature_noarea_mosi68_20261025`. Source5fb3659 is a separate immutable overlay;
the original4d5aa09 numerical source and all completed comparisons remain intact.
LAUNCH.json contains the actual command,UUIDmap and every deployed Python hash.

Local and remote17focused tests passed. A real MOSI rate0.7 two-update CUDA smoke
passed in9.015seconds,peak377.515MiB allocated GPU memory,2.338GiB peakRSS.
Initialization,parent and saved-mask replay were exact; frozen-parent gradients
remained absent; actual masks/update traces matched the same-RNG original smoke.
Smoke is implementation evidence,not a training performance result.

At the first eight-worker check,all8formal jobs were running (6–45/150epochs),
with no errors. PhysicalGPU0/1/2 carry3/3/2workers; no other cards/tasks altered.
Only8new seed68 branches,not extra seeds or a repeated Flat/full Signature.

Outputs: `/data2/yb/remote_experiments/osram_signature_noarea_mosi68_20261025/execution_v1`.
Controller log: sibling`controller_v1.log`. QUEUE.json and per-job SUMMARY.json
are the live status authority. Paired ACC/WF1 report is automatically generated
after all8jobs complete and selected artifacts/traces validate.
