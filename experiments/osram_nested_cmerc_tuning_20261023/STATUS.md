# Launch receipt — not final scores

INTERNAL TEST-ORACLE SCREEN. Snapshot: 2026-10-23 13:51 UTC.

Finite21-job queue is actually running on biggpu physicalGPU7, persistent tmux
`nested_cmerc_tuning_20261023`, controllerPID3725311. Initial live children:
MOSEI FlatPID3725708 and IEMOCAP4 fold5 FlatPID3726264. Six Nested variants per
task and IEMOCAP6 control are pending automatic resource admission. This is NOT
a claim that all21 jobs are concurrently training or have finished.

No MOSI/MELD jobs or new modules are added. Six existing variants + one matched
Flat per task; IEMOCAP4/6 initial screening uses seed66/fold5 only. At most4
concurrent workers; initial headroom admitted2. Measured long-dialogue smoke
peak3.433GiB; reserve8GiB per worker plus2GiB margin, without killing other jobs.

23 protocol/source/parity tests passed locally and in the actual cmerc environment.
Largest-variant real-data smokes: MOSEI3 optimizer updates in71.47s, peak813MiB;
IEMOCAP6 three updates in70.73s, peak3515MiB. Both verified finite/nonzero
tokenizer/GIN/pool/readout/decoder task gradients, one Memory scan per update,
Adam/model/CPU+CUDA RNG saving and exact selected-prediction reload. These timings
include setup/evaluation/checkpoint work and are NOT formal training benchmarks.

MOSEI smoke used first sourceb9af9a9; IEMOCAP retry and ALL full training use3777ab1.
The intervening changes add archive-manifest verification and correct a validation
assertion only; original numerical model, loss and sampler source did not change.
First IEMOCAP smoke failed a bitwise comparison and remains preserved as failed,
not silently erased. Equal original parameters/Flat inputs and exact zero Nested
correction were verified; logits differed by float32 roundoff and no class changed.
Numerical parity now records its true maximum error; see README for the boundary.

Same-source matched Flat controls are intentional: historical IEMOCAP test masks
were resampled by epoch, whereas this screen replays accepted exact selected
masks. Uniform UUID CUDA RNG repair also differs from the old MOSEI cmerc runner.
Neither old score is presented as an exact training-trajectory comparator.

All full runs are from scratch. Original CHM/Query/KV/write/read-before-write and
Flat1600 remain unchanged. Completed weights/masks/history are protected. Every
run saves8BEST plus matching predictions, exact masks and resumable last.pt.
The controller validates completion then writes RESULT.md/json/PER_RATE.csv;
it will not launch extra seeds, folds or an unlimited search.

SourceSHA, remote paths, smoke hashes and launch metadata: LAUNCH.json.
Treat this file as a launch snapshot; live progress is QUEUE.json/SUMMARY.json.
