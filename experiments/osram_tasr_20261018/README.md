# Trajectory-Aware Source Reading (TASR)

INTERNAL DIAGNOSTIC ONLY. Finite seed66 screening, not a performance claim.
Design/algebra/control semantics: [DESIGN.md](DESIGN.md).
Pre-screen tests/real GPU evidence: [VERIFICATION.md](VERIFICATION.md).

## Implementation

- `gcnet_missing_m3/tasr.py`: additive Value-slot tracking, strictly past
  count/coverage/age/ever-seen, shared target/source adapter.
- `gcnet_missing_m3/osram.py`: opt-in reuse of ORIGINAL joint weighted solve;
  sources decay/read/update beside the authoritative shared Memory. Correct
  only missing-target forward Gap before original Flat. No new query/write.
- Config/CLI: `--osram-source-tracking` for parameter-free inspection;
  `--osram-tasr --osram-tasr-source-mode real|neutral` for correction.
  Defaults off; original states/checkpoints/RNG unchanged.
- Adapter59,423 parameters. Source tracking has zero parameters. Three sources
  denote coupled joint-write Value-slot contributions, NOT pure modalities.

## Frozen screening commands

Accepted MELD lr1e-5/wd1e-5 Flat and original IEMOCAP6 seed66/fold5 Flat are
resolved from protected registries, all hashes verified. Each dataset uses its
ONE original miss=.3 checkpoint for every condition. That original checkpoint
was chosen with Test-oracle; new adapters use validation only. Do NOT compare
old per-rate best-checkpoint means to this fixed-anchor F0 as identical scores.

Runtime: `ssh biggpu`, `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Exactly four jobs,30epochs,Adam3e-4,wd1e-5,batch32; only Adapter trainable.
Each step=.8 original GCNet random-view CE+.2 independent Persistent-view CE;
the second view starts from complete raw data, not a subset of the first.
No contrastive/auxiliary loss. All original modules frozen in eval mode.

```bash
python -m experiments.osram_tasr_20261018.launch \
  --output /home/qinjinghui/remote_experiments/osram_tasr_20261018/runs_v1 \
  --source-commit 38251f4b5ca1f1a64ea698eb3cc3a4fb9acf2a65
```

Actual immutable training checkout:
`/home/qinjinghui/remote_experiments/osram_tasr_20261018/source_38251f4`.
Persistent runner: tmux session `tasr_38251f4`; launcher log
`/home/qinjinghui/remote_experiments/osram_tasr_20261018/launch_38251f4.log`.
GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`, up to4 jobs with3GiB/job
reservation based on .31/1.02GiB real frozen smoke peaks. Never physicalGPU4;
busy GPUs/other processes are untouched. Queue state/commands/PIDs/logs:
`runs_v1/queue/QUEUE.json`. Initial four PIDs:2216777/2216926/2217017/2217020.

Individual job (fresh output; optional `--resume` restores an existing last.pt):

```bash
python -m experiments.osram_tasr_20261018.run --phase train \
  --dataset MELD --mode real --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
  --output /absolute/new/job/output
```

Other allowed dataset: `IEMOCAPSix` (seed66/fold5 only), mode `neutral` for F1.
MELD uses official dev; IEMOCAP official loader aliases val/test, so Adapter
instead holds out deterministic10% of original TRAINING conversations. The
frozen original backbone saw that holdout previously: Adapter-only validation,
not a clean end-to-end validation split. Test-fold labels never select adapters.
Validation uses fixed/replayed original GCNet masks across8 rates; maximize
mean ACC, then lower meanCE. One selected Adapter for all test conditions.

## Outputs and verification

`best_adapter.pt`/`last.pt` contain Adapter, optimizer, epoch, RNG and immutable
anchor/config/source identity; no duplicate13M-parameter backbone is saved.
Restore by strictly loading original anchor then Adapter/optimizer/RNG. Output
directories are independent; protected checkpoints/predictions remain intact.

Each job retains `history.json`, `SUMMARY.json`, selected epoch, validation-mask
hashes, random training-mask traces, Persistent pattern counts, timing/peak
memory and paired `evaluation/RESULTS.json`. Large raw predictions/masks stay
remote in NPZ. Current-code strict load and zero-on logits must reproduce the
original protected .3 cohort exactly, or training refuses to start.

Final evaluation uses protected exact selected mask files ONLY as common input
masks, with metadata alignment checked. F0 and adapted model use the same .3
anchor at ALL rates and all seven Persistent patterns. F0/Adapter share labels,
speaker/utterance order and availability. Report per-condition ACC/UA/WF1/CE,
corrections/harms, per-class recall/support, source errors, gamma/trajectory/
Gap norm statistics; compare F2 to both F0 and same-capacity F1.
No test-best epoch or mask is newly selected during this screening.

Core regression command:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest \
  tests/test_tasr_integration.py tests/test_tasr_primitives.py \
  tests/test_tasr_protocol.py tests/test_tasr_runner.py -q
```

Real smoke (fresh independent output) is also reproducible:

```bash
python -m experiments.osram_tasr_20261018.smoke --dataset MELD \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 --output /absolute/new/smoke
```

Do not expand LR/network/seeds/folds if F2 fails to exceed F1. Gamma is bounded
but sources may cancel: a .25 coefficient bound does not bound the correction
to .25 of the original Gap norm. Diagnostics are not utility/semantic proofs.
