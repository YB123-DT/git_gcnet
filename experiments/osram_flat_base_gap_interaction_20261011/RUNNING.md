# Frozen diagnostic execution

- Server: biggpu; physical GPU6, UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
- Immutable execution code: 8945717; source archived before launch.
- Remote root: /data1/yb/remote_experiments/osram_flat_base_gap_interaction_20261011.
- tmux socket: flat_bg_20261011; session: diagnostic; launch Python PID: 1623930.
- Log: remote root/run.log; output: remote root/results.
- Source checkpoints: /data1/yb/remote_experiments/osram_nested_training_gradients_20261010/runs/flat_seed66.
- No training. Existing capacity runs and Rooted Probe not changed.
- Local and remote arithmetic/filter tests: 2 passed. Full archive parity and
  frozen state/file hashes checked separately for every missing rate.
- Completed eight rates, 5488 valid rate/utterance records. Local independent
  CSV re-computation agrees with remote summaries within 1e-12. Exact Python
  float equality initially failed for W-F1 differences of about 1e-14 across
  environments; counts agree exactly. No predictions or metrics were changed.
