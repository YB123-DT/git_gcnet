# Verified launch: original Flat + B3

INTERNAL TEST-ORACLE SCREEN. Training is running; no final scores are available yet.

- Server: `ssh biggpu`; physical GPU2 only, UUID
  `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`.
- Immutable training source: `737fdb16e5b4f4ee7a426a0979efbc7f59c5ed8c`.
- Source checkout: `/home/qinjinghui/remote_experiments/osram_meld_b3_code_737fdb1`.
- Output: `/home/qinjinghui/remote_experiments/osram_meld_b3_structure_hpo_20261016`.
- Persistent tmux session: `meld_b3_gpu2_1016`; controller PID at launch: `4150473`.
- Python: `/data2/yb/conda_envs/cmerc-repro/bin/python`;
  Torch `1.13.1+cu117`, Optuna `5.0.0`.
- Dataset SHA256 verified:
  `a0349614b2b58e8b2db768f0144aa36b96fb2348ca436a6879ee2623cc2d2f80`.

The queue verified five concurrent worker processes (not five serial jobs):

| Trial | Tokens | Rounds | Added B3 parameters | Launch PID |
|---|---:|---:|---:|---:|
| 0 | 4 | 2 | 2,188,416 | 4154946 |
| 1 | 2 | 2 | 2,188,160 | 4154987 |
| 2 | 8 | 2 | 2,188,928 | 4155102 |
| 3 | 4 | 1 | 1,526,016 | 4155335 |
| 4 | 4 | 3 | 2,850,816 | 4155876 |

Each worker trains MELD seed66 for 50 epochs, batch32, lr0.001, WD1e-5,
dropout0.5, original Flat1600 and Local Skip, cyclic0.0–0.7,
`gcnet-random-mask-v1`. The existing MBT block updates task-side Local/Base/Gap;
raw Local Skip and all CHM equations are unchanged.

After all five initial jobs finish, fifteen additional unique TPE configurations
start automatically, at most five pending/concurrent. Total budget is twenty,
with token count1–8 and rounds1–4; internal width128 and training hyperparameters
remain fixed. SQLite state, exact effective configs, hashes and process identities
are retained remotely. Other GPU tasks, including GPU5 IEMOCAP runs, were untouched.

## Verification

- Local targeted suite: 34 passed; two Optuna tests skipped only because the local
  environment lacks Optuna. Remote installed-environment suite: **18 passed, none
  skipped**, including real SQLite proposal/feedback/uniqueness/resume tests.
- Compared against the actual pre-change `09bf963` representation implementation:
  default B3 has identical state tensors, RNG state and outputs with nonzero
  decoders. Tests additionally cover Flat-off equivalence, zero-init, first-turn
  bypass, inactive Gap/padding safety, original Local Skip and delayed gradients.
- One largest searched structure (8tokens, 4rounds) real MELD batch, three actual
  backward/optimizer updates: complete; peak1886.161MiB, 18.272seconds.
  Nonzero Block2 gradient tensor counts by step: 0,18,343. This explicitly checks
  learning beyond the initial zero decoder; it is not performance evidence.
- `py_compile` and `git diff --check` passed. The existing optional torch-sparse
  shared-library warning is recorded; the model does not use that dependency.

## Evaluation boundary

Per-rate Test-best ACC checkpoints keep their exact selected predictions/masks;
UA must come from those same predictions. Fixed common-mask paired replay is
reported separately. TPE uses the pre-registered paired objective from README.
The paired Flat reference is the retained same-hyperparameter seed66 screen
(paired ACC60.603448, UA35.114037), not the historical three-seed mean60.693.
No historical B3 score is counted as a newly completed run.

Run the command in README from the immutable checkout to resume this exact queue.
Do not deploy later numerical changes into this running checkout or treat a
running trial as a completed result.
