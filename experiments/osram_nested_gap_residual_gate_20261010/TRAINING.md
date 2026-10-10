# Authorized MOSI three-seed run

INTERNAL DIAGNOSTIC ONLY — inherited per-rate Test-oracle protocol.

User authorized formal execution after implementation on 2026-10-10. Run seeds
66/67/68 for100 epochs, eight cyclic missing rates, using each seed's original
cfg84 Flat configuration as the protocol reference. The only configuration change
is `osram_meaningful_block= nested_gnn_gap_residual_gate`. Architecture is the
original Nested plus the approved decoded Gap residual Gate. No baseline rerun,
hyperparameter sweep, persistent mix, auxiliary loss or staged/frozen training.

Reuse the existing source-pinned experiment runner, dataset manifest hashing,
complete epoch-boundary TrainingState and per-rate checkpoint/prediction checks.
Each output saves `last_training.pt`, eight `best_miss_0p*.pt`, predictions, effective
config, history and PROVENANCE.json. Training versions remain retained by the
existing recovery code; no old checkpoint cleanup is performed.

Server biggpu, planned physical GPU7 UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`; GPU4 excluded. Launch independently
in persistent tmux, check actual resource admission before each additional seed.
Use an immutable git-archive snapshot, not the dirty local worktree. Latest GPU7
measurement before setup: 0MiB used,32495MiB free. Storage27GiB free; estimate
up to approximately19GiB for all three runs including recovery versions.

Entry point:

```text
python -u -m experiments.osram_nested_gap_residual_gate_20261010.run
  --seed SEED
  --reference /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_SEED/config.json
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
  --output /data2/yb/remote_experiments/osram_nested_gap_residual_gate_20261010/runs/seed_SEED
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
```

The actual PID, command, source commit, log, tmux session and launch observations
are recorded in LAUNCH.json after startup. A launch claim requires a live process
and model/training initialization evidence, not only a returned shell command.
