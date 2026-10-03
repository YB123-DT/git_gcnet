# Implementation verification, before remote admission

INTERNAL DIAGNOSTIC ONLY. No meaningful20 training result is available yet.

The research/design record is committed as `7060222`; implementation is a
separate commit. The matched seed66 Flat reference is 81.06809539495711%
eight-rate mean W-F1 and 76.35225088637502% high-missing mean. Its historical
training commit was not recorded: the source hashes and protocol in
`BASELINE_AUDIT.json` are the evidence, not a guessed commit identifier.

## Implemented boundary

`--osram-meaningful-block none` is the unchanged default. Twenty separate
source-grounded cores are available behind this independent flag. Each uses
current Local and existing forward Base/Gap reads; a zero-initialized output
projection adds its result before the original Flat emotion normalization.
There is no new OSRAM query, write, trajectory, task head, or auxiliary loss.
Inactive evidence is sanitized before projections, and first-valid utterances
and padding skip the branch. Initializing a branch is RNG-isolated; Slot
Attention's training noise and NODE's data initialization use private state.

The implementation is a task-specific adaptation, not a reproduction claim
for the original papers. Primary sources, transferred cores, and necessary
departures are recorded in the six source-card files and independent reviews.

## Verification observed so far

```sh
CUDA_VISIBLE_DEVICES='' /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p 'test_meaningful*.py'
```

Initial observation: 90 tests passed (13.673 seconds), including all twenty full-model
train/eval zero-bridge output and global RNG comparisons, masking/core
gradient tests, and interrupted/resumed real tiny-trainer equivalence.
This is CPU correctness evidence, not CUDA readiness or evidence of gains.
After the data-binding, queue/recovery, CUDA-checker and float64 grouping
corrections, the same command passed **106 tests in 13.404 seconds**.

Independent source review corrected NODE leaf numbering and the Sheaf
normalization operator; neither initial implementation entered formal training.

## Existing regression-suite limitations

The seven `tests.test_readout_candidate_integration` tests passed.
`tests.test_osram_per_rate_selection` cannot import because pytest is absent
from the existing environment; no dependency was installed.
The two `tests.test_miss0_parity` assertions also fail on an untouched export
of pre-change commit `95e7dad`. They are recorded as pre-existing failures,
not silently fixed or counted as passing here.

CUDA admission, full-epoch throughput, all twenty training runs, three-seed
confirmation, and performance comparisons remain pending. Do not call the
experiment complete on the strength of these tests.
