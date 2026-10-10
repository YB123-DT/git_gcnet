# Original Nested with only Gap-T residual gating

INTERNAL DIAGNOSTIC ONLY

User authorized the Gap-T-only experiment after the completed all-Gap Gate run.
No score is available at implementation time; actual launch status belongs in
LAUNCH.json and each remote PROVENANCE.json.

```
Gap-A_out = original Gap-A + Nested delta Gap-A
Gap-T_out = original Gap-T + gate-T * Nested delta Gap-T
Gap-V_out = original Gap-V + Nested delta Gap-V
```

Original Local and Base corrections remain unchanged. One scalar T Gate shared
across its eight64d heads; same211→32→1 gate MLP, same64d projections and type
embedding as the preceding all-Gap variant. The final gate layer is zero-init,
giving active T gate1; A/V active residual coefficients are hard-fixed1. Inactive
coefficients/output contributions are0. Only current T-missing rows contribute
task gradients to the gate. No gold label or historical sentiment enters it.

New method: `nested_gnn_gap_t_residual_gate`. No new loss, freeze, staged training,
JEPA, completion, query or Memory modification. Unchanged cyclic random missing,
MOSI seeds66/67/68,100epochs, batch32, Adam.001, weight_decay1e-5, MSE and per-rate
Test-oracle selection. Each seed trains once across the cyclic rates and saves
eight selected checkpoints. Existing original Flat/Nested and all-Gap results
are reused, not retrained.

Gate parameter budget remains89,399, exactly matching the all-Gap Gate; some
non-T type entries are unused by the gate. This is not a newly compressed Gate.
Original parameter initialization and downstream RNG are preserved.

Checks: tests written before implementation; all28 new/existing targeted tests
passed in6.85s, one existing pynvml environment warning. Exact original equivalence
with nonzero Nested decoders, .5T-only intervention, no T-present task gate
gradient, finite T-missing updates and capacity/RNG checks passed. Python compilation
and diff check passed. No repeated GPU smoke is performed before actual training.

Server remains biggpu, healthy physical GPU7, UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`. Fresh source/log/output root is
`/data1/yb/remote_experiments/osram_nested_gap_t_residual_gate_20261010` because
data1 has179GB free versus19GB ondata2. Original data/reference configurations
stay ondata2. No old experiment data or checkpoints were removed.

Reused training runner verifies immutable source, dataset hashes, evaluation masks,
100epochs and all20 selected/recovery artifacts before marking complete. Keep
original eight BEST checkpoints and last_training.pt plus recovery versions.

Entrypoint:

```
python -u -m experiments.osram_nested_gap_t_residual_gate_20261010.run
  --seed SEED
  --reference /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_SEED/config.json
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
  --output /data1/yb/remote_experiments/osram_nested_gap_t_residual_gate_20261010/runs/seed_SEED
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

Baseline three-seed references: original Nested80.489 eight-rate/75.252 high;
all-Gap Gate80.225/75.449; original Flat80.559/75.594. These are internal protocol
references, not expectations that the T-only run will outperform them.
