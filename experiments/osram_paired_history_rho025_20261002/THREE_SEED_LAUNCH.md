# Three-seed follow-up — running

Prespecified at user authorization:rho .05,.10,.25. Rationale:.10 highest seed66
eight-rate mean;.05 highest high-missing result among the scanned configurations;
.25 user-proposed3:1 weighting. This selection used exploratory Test-oracle
results; the follow-up does not turn it into formal validation-selected evidence.

Reuse completed seed66; train only seeds67/68, six new100epoch runs total.
GPU5:three seed67 jobs; GPU6:three seed68 jobs. Independent output directories.
Original config read separately for each seed; only paired-view flags and rho
added. No gate/query adaptation, no InfoNCE contribution, no backbone changes.

Remote root:`/data2/yb/remote_experiments/osram_paired_rho_three_seed_20261002`.
Isolated code commit97474eb. Coordinator PID278627, persistent independent session.
Its log is coordinator.log; runs/children.json updates every30seconds after launch.
Each child writes config, source hashes, provenance, history, per-rate BEST weights.
Coordinator checks process exits and produces runs/SUMMARY.json only after all
six new jobs succeed; errors recorded in status.json. No automatic blind retries.

Preflight:both cards empty, UUID checks;622GB free on /data2; all six configs pass
TrainConfig validation, contrast weight0, query adapter disabled. Script compiles.
Final reporting must compare all nine seed/rho results to seed-matched Flat and
show mean/std and per-seed results, not just the best run. Training is not complete
at this handoff. Persistent monitoring remains server-side after chat returns.
