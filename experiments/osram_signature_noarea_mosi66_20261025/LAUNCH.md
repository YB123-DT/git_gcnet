# Verified seed66 launch

Corrected source6314804/execution_v2 launched on biggpu at
2026-10-25T16:06:19Z in tmux`signature_noarea_mosi66_20261025`.
LAUNCH.json records numerical cwd,overlay hashes,actual command and GPU UUIDs.
Original numerical sourcee0ca94c and existing normal Signature/Flat untouched.

Local and remote18focused tests passed. Real MOSI rate0.7 two-update smoke
passed in9.542seconds,peak377.515MiB CUDA allocation and2.327GiB processRSS.
Initialization,prediction/mask replay,frozen parent and original same-RNG
training-trace checks passed. This is implementation evidence only.

All8formal jobs were running on physicalGPU0/1/2 at the first complete admission
check,46–90/150epochs,no errors. Only frozen-parent NoArea branch parameters
train; no new Flat,full Signature,other seeds or hyperparameter searches.
Controller automatically validates and summarizes after all8complete.

Outputs:
`/data2/yb/remote_experiments/osram_signature_noarea_mosi66_20261025/execution_v2`.
Log: sibling`controller_v2.log`. Original failed import preflight remains under
source68ab183/execution_v1/controller_v1.log and is not part of the score table.
No other tasks were signalled or altered.
