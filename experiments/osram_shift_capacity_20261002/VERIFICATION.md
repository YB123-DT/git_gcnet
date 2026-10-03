# Capacity variant verification and launch

Implementation commit04aec16. Four configurations are D2-W128, D2-W256,
D3-W128, D3-W256. Seed66,100epochs; relation128, scalar outputs1 unchanged.
Default depth1/width128 preserves old parameter keys, initial weights and RNG.
Filter MLP alone changes; old Flat/trainable skip/adapter, Memory, mask and loss
remain original single-view no-JEPA. No paired-view or contrastive objective.

Remote GPU5 V100 verification:20 tests passed, including CUDA exact initial
Flat logits and RNG compatibility. Tests cover inactive/padding safety, active
denominator, zero-init residual, finite gradients and all requested dimensions.
Four effective configs checked against original cfg84; only readout selection
and depth/width differ. Constructed layers match all four documented paths.

Local CPU suite passes (CUDA test skipped). On local CUDA hardware, the old
bitexact CUDA test also fails with pre-change code at sub-micro differences;
this was not hidden by changing tolerances. The intended remote GPU5 passes
the unchanged test. No claimed cross-hardware bitwise equivalence.

Server biggpu, isolated root:
`/data2/yb/remote_experiments/osram_shift_capacity_20261002`.
Coordinator PID1403586, persistent session; log coordinator.log.
Each of four children runs on hostGPU5 with its own outputs/log/provenance;
starts stagger15seconds for resource checks, training is concurrent, not queued.
GPU6 is occupied by unrelated work; no unrelated process stopped or batch altered.
Coordinator checks child exit status every30seconds and records failures.
Per-rate BEST checkpoints retained; originals reused. Initial handoff was running.

Completion checked2026-10-03: coordinator complete, all four runs complete100epochs,
each retains8BEST weights. Local results/ archives configs, provenance, histories,
metrics and process status; summarize.py reconstructs all four eight-rate and
high-missing means from the per-rate BEST metrics, never the last epoch.
