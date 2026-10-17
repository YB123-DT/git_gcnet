# MELD frozen Memory parameter and head sensitivity

INTERNAL DIAGNOSTIC ONLY. No training, no new model, no mask generation.
Static trained parameters for all24protected BESTs are in RESULT.md. The new
functional diagnostic is preregistered in SENSITIVITY_PLAN.md and is complete,
reported separately in SENSITIVITY_RESULT.md. Three rates are not an eight-rate result.

## Version and execution

- Frozen driver commit:801f4a29bc02ac29528d428d5b4f78738a59c85e.
- Numerical model and loader commit:b4a313841f098487a6482d2712bdac94aa8e319e.
- Server:biggpu; Python:/data2/yb/conda_envs/cmerc-repro/bin/python.
- Host GPU3,UUID GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a; logical cuda:0.
- tmux:meld_memory_sensitivity_801; initial Python PID2575407.
- Remote output:/data2/yb/remote_experiments/osram_meld_memory_sensitivity_20261017/frozen_801.
- Adjacent log:frozen_801.log. The runner writes only to its isolated output;
  protected checkpoints, masks and feature files are opened for reading only.

Completed2026-10-17 in328.86seconds. SUMMARY.json status=complete,
all2610utterances per rate, exact original saved-label/class parity, matching
sample identities across rates, restored state hashes and unchanged CUDA RNG
were verified.
Do not infer completion from the tmux/session or process exit alone.

```bash
cd /data2/yb/remote_experiments/osram_meld_memory_sensitivity_20261017/driver_801f4a2
CUDA_VISIBLE_DEVICES=GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  experiments/osram_meld_memory_sensitivity_20261017/run.py \
  --source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline \
  --mask-root /data2/yb/remote_experiments/osram_meld_testbest_ua_20261014/exact/source_0 \
  --model-source /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138 \
  --dataset-root /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138/dataset \
  --meld-pickle /data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --output /data2/yb/remote_experiments/osram_meld_memory_sensitivity_20261017/frozen_801 \
  --gpu-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

The runner refuses an existing output directory; a new intentional replay must
use a distinct output path. It rejects final-mask roots, altered model source,
wrong feature hash, wrong CHM settings and noncanonical missing masks. It uses
temporary logit overrides only inside no-grad eval and restores the Parameter
objects bit-for-bit. No original mathematical function is edited.

Torch1.13's NVML UUID count parser can disagree with CUDA runtime. The runner
asserts a single CUDA runtime device and synchronizes both Python device-count
aliases before model initialization; it does not change the environment, GPU
binding or source model. A torch-sparse optional-library import warning occurs
in this environment; this OSRAM path does not use it, and original prediction
parity is an independent required check.

CPU loader iteration consumes its ordinary base-seed RNG; paired conditions
reuse the same materialized batch and masks and consume no CUDA randomness.
No claim of whole-process CPU RNG invariance is made. All model parameters,
including temporarily overridden alpha/beta, must be restored.

## Checks

```bash
/data2/yb/conda_envs/cmerc-xp-repro/bin/python -m pytest -q \
  tests/test_meld_memory_sensitivity.py tests/test_meld_sensitivity_runner.py \
  tests/test_meld_fixed_query_replay.py tests/test_meld_write_retention.py \
  tests/test_meld_gcnet_random_no_skip_evidence.py
```

24tests passed before deployment. They cover temporary-parameter restoration,
identical Local/Query/K/V, first-turn causality, inactive/padded evidence,
active-column spectra, embedded-beta/ridge equivalence, joint-write column
permutation, exact forward head slicing, learned-alpha fork parity, identities,
fixed seven-class UA and prior mask/task replay contracts. They do not prove
performance gains or rule out all scientific failure modes.

After adding artifact/noise-bound checks and the first-source alpha identity,
the complete suite above plus tests/test_meld_sensitivity_analysis.py passed
30tests in32.86seconds. The actual frozen run completed with all original
saved classes/labels matched, state hashes restored and CUDA RNG unchanged.
Noise-aware postprocessing is at the adjacent remote analysis_noiseaware;
the earlier analysis_724 head-correlation output is superseded as documented
in SENSITIVITY_RESULT.md. Reproduce postprocessing with:

```bash
python experiments/osram_meld_memory_sensitivity_20261017/analyze.py \
  --input /data2/yb/remote_experiments/osram_meld_memory_sensitivity_20261017/frozen_801 \
  --output NEW_NONEXISTING_ANALYSIS_DIRECTORY
```

Raw logits/IDs, large utterance/state/spectrum/direct-beta/lagged-alpha gzip
streams remain remote. Only compact summaries, source/tests and text reports
are committed. Readout-only head removals retain original Memory; they measure
marginal use in this trained decision path, not unique semantic roles or
whether a head can be pruned and retrained without loss.
