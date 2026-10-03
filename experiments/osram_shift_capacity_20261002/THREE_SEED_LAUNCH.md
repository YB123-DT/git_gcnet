# D3-W256 three-seed follow-up

User selected768→256→256→256→1. Reuse completed seed66; only seeds67/68
are newly trained,100epochs each. Each seed reads its own original Flat config;
only memory-shift-residual selection and filter depth3/width256 are added.
Relation128, scalar output, zero-init residual, single-view random-missing task
training remain unchanged. No paired views, InfoNCE, completion, or query changes.

Remote biggpu hostGPU5, two concurrent independent persistent training processes.
GPU4 excluded; assigned card checked empty before launch. No batch changes.
Root:`/data2/yb/remote_experiments/osram_shift_d3w256_three_seed_20261003`.
Code isolated in code/, outputs in runs/, logs seed_67.log and seed_68.log.
Code commitbc1ab02. Seed67 PID441154; seed68 PID441155. launch.json contains commands.
Each run records its own config, source hashes, provenance and eight BEST checkpoints.

Verification:20 local tests (19pass,1CUDA skip); existing remote GPU5 capacity
suite previously passed20/20. Both remote seed configs passed schema validation
and exact difference checks against seed-specific original Flat configurations.

Status at launch:running, no new result yet. Final comparison must use seeds66–68,
per-rate BEST, both eight-rate and high-missing mean, standard deviation and
paired-seed deltas against Flat. Internal Test-oracle diagnostics, not formal
validation-selected results; this variant was selected after a seed66 capacity scan.
