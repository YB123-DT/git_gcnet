# Launch verified

INTERNAL DIAGNOSTIC ONLY. Training is running; no final performance result yet.

2026-10-11: both tasks completed their first train/evaluation epoch, PIDs alive, JEPA loss0. No extra seeds, datasets, thresholds or learning-rate searches launched.

| Task | Seed | Budget | GPU | PID | Verified epochs |
|---|---:|---|---:|---:|---:|
|nested_rooted_concat|66|100@.001 +50@.0001|1|3014513|1|
|nested_concat_capacity_control|66|100@.001 +50@.0001|1|3015457|1|

Server biggpu; healthy GPU1 UUID GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049. Concurrent, not sequential. Controller3014507; all processes detached via start_new_session. GPU4 excluded and no newGPU7 jobs. At verification GPU1 allocated3804MiB. Original raw feature dimensions512/1024/1024 and2199 total records match the existing loader.

Immutable code snapshot a699d56, /home/qinjinghui/yb_gcnet_experiments/osram_nested_concat_20261011/source_a699d56. Only committed Python files deployed; unrelated local gcnet/model.py modifications were not included. MANIFEST.json records source/data/config hashes, environment and two exact config deltas. LAUNCH.json records command/controller. Full-state training and epoch100→150 continuation reuse experiments/osram_complete150_20261011/run.py. Each task saves isolated stage100/stage150 with BEST/predictions/full recovery states and LR traces. Stage100 config says100 by design; the controller then resumes to150 at1e-4.

## Implementation verification

Remote immutable snapshot: `python -m pytest -q tests/test_nested_concat.py tests/test_memory_shift_residual.py` →9 passed. New module4 tests verify canonical pre-decoder core output reuse, masks/NaNs/padding, first-turn zeroZ, original initialization/RNG/output and Memory trajectory equivalence, capacity matching, finite gradients and actual graph updates over3 optimizer steps.

Broader local run:16 passed,20 subtests passed,1 failed in existing test_nested_sweep.py::test_each_ablation_changes_only_its_named_mechanism. That legacy test asserts the entire NESTED_SWEEP tuple equals its older short METHODS list; the registry already contained12 later variants before this task. Neither that test nor NESTED_SWEEP was changed by a699d56. Do not claim full-suite green. Local pynvml deprecation warning is environmental, not changed.

| OSRAM-only counts | Params | Adapter hidden |
|---|---:|---:|
|Original Flat|12578912|1600|
|Original Nested|12738147|1600|
|Concat|13200867|1600|
|Capacity control|13198024|1704|

Concat adds621955: graph109315 plus expanded adapter512640. The original159235-param Nested includes decoders that are absent here. Concat/control differ by2843 parameters, not exactly equal. Whole-model totals should be recorded from final metrics; these are measured OSRAM-only counts, not mislabeled full-model counts.

Raw Flat input4352 + graphZ320 =4672; final hidden1600 and original LocalSkip/head retained. Z is the core.readout pre-decoder output, NOT the internal rooted-only pool. No delta-evidence reconstruction, auxiliary loss or new Memory computation.

Remote root /home/qinjinghui/yb_gcnet_experiments/osram_nested_concat_20261011; logs/{method}.log and runs/{method}/PROVENANCE.json. Existing Flat66 lower-LR150 (81.068 W-F1) and original Nested66 lower-LR150 (81.163) are references, not freshly rerun controls. Do not infer gains from the first epoch.
