# Actual execution record

Source implementation commit: `64b9b551832079a7417aef4261b19a2d27bb097a`.
Source bundle:159 files,426649 bytes; archive SHA256
`854998306652e8850d0c56a0e508a9fc185e4505492444eb60c7f6b5893c841f`.
Native model/sampler files were not edited.

Local checks:48 focused/native CPU tests passed (5.75s); independent reviewer
reran the26 unified tests successfully. Compileall and staged whitespace checks
were run. GPU verification is a separate gate; CPU tests are not performance.

Server:biggpu. Python:`/data2/yb/conda_envs/cmerc-repro/bin/python`.
Source:`/data2/yb/remote_experiments/cam_unified_20261030/source_64b9b55`.
Outputs:`/data2/yb/remote_experiments/cam_unified_20261030/execution_v1`.

Resource precheck:363GiB RAM available,499GiB disk free. GPUs0–6 were occupied;
physicalGPU4 remains forbidden. PhysicalGPU7 UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e` had about29GiB free. Existing tasks
were not stopped or changed.

Preflight started in tmux:`cam_preflight_20261030`, initialPython PID2713072.
Log:`execution_v1/preflight.log`; evidence:`execution_v1/PREFLIGHT.json`.
Preflight passed in39.57s. The protected epoch17/miss0.3 checkpoint reproduced
all4659 test predictions and metadata bit-for-bit (22causal scans). ACC84.755091,
WF1=84.610594, MSE0.57229185 are **compatibility results**, not new training scores.
Zero-init original Nested also matched exactly. All original expected projector
gradient groups became nonzero after startup; all checked gradients were finite.
The longest real batch contained32 conversations with padded length98.
Peak allocated2214.26MiB, reserved2388MiB; per-job admission budget4096MiB.

The four MOSEI jobs started in tmux`cam_ablation_20261030` on physicalGPU7:
A0 PID2716574, A1 PID2716575, A2 PID2716576, A3 PID2716577.
Each is from scratch,60epochs,seed66,constant1e-4. No old weights initialized
training. QUEUE.json and the separate SUMMARY/history files are authoritative.

Latest user update: **only IEMOCAP-6 and MOSEI**. Keep the four running MOSEI
jobs untouched; extend the unified package for original-protocol IEMOCAP-6
seed66/fivefold A0–A3, using100epochs/1e-3 and ACC-selected matching UA/WF1.
No MOSI, MELD, extra seed, HPO or second-stage GNN training is authorized here.
IEMOCAP implementation/preflight is pending at the time of this record.

Recovery check verifies state loading and CUDA RNG coverage; numerical equality
of the next optimizer update after interruption has not been independently
demonstrated. Do not overstate that verification.
