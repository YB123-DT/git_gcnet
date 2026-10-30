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
IEMOCAP implementation is pinned to source commit
`152a91eb1882796eac3ac827f8377c721d97086e`, a separate165-file441189-byte bundle
with SHA256`1e9167a3e44d3492ce06364767d1ad762f91af8a364108be93f2fa06ff32a547`.
Source:`/data2/yb/remote_experiments/cam_unified_20261030/source_152a91e`.
Outputs:`/data2/yb/remote_experiments/cam_unified_20261030/iemocap_execution_v1`.

IEMOCAP GPU preflight passed in33.60s: protected fold5/miss0.3/epoch69 all1623
test logits/predictions/metadata matched bit-for-bit. ACC64.325323,UA63.389523,
WF1=64.126752 are replay compatibility numbers, not new ablation scores.
The real longest training batch contained32 conversations with padded length110;
CE/backward/update and state restoration passed. Peak allocated2032.21MiB,
reserved2214MiB; admission budget3840MiB per training process.

The finite20-job IEMOCAP queue is running in tmux`cam_iemocap_20261030`.
It processes fold5 first, thenfold1–4; each fold has A0–A3. It subtracts live
MOSEI workers from the shared per-GPU limit4 and waits for slots instead of
launching an additional4 processes over them. At queue launch all4MOSEI
processes were alive, so IEMOCAP was **queued, not yet formally training**.
Only GPU7 was available; no other person's job was stopped or migrated.
The queue automatically summarizes every fold/rate when complete; no further
interactive command is required to advance its approved finite scope.

Latest local verification:64 focused/native tests passed (5.71s), plus independent
14-test IEMOCAP review. The native model/sampler packages remain unmodified.
The environment emitted an optional torch-sparse import warning about
libcusparse.so.11; no environment was changed to suppress it. Both original
checkpoint prediction replays were exact on the actual OSRAM path.

Recovery check verifies state loading and CUDA RNG coverage; numerical equality
of the next optimizer update after interruption has not been independently
demonstrated. Do not overstate that verification.
