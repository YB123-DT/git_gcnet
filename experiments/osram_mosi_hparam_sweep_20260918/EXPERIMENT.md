# MOSI causal OSRAM hyperparameter screening

This queue is an internal diagnostic, not a paper result. It keeps the
causal OSRAM/no-JEPA control fixed:

- cyclic missing-rate training;
- forward-only OSRAM with `osram_write_step=0.6`;
- mean fusion and Flat emotion readout;
- the inherited frozen input features and masks;
- independent Test-oracle checkpoint selection for each missing rate.

The 90 configurations are a coarse, reproducible coverage of the requested
learning-rate, batch-size, optimizer, weight-decay, dropout, projector
dropout, block learning-rate multipliers, clipping, schedule, epoch, and
OSRAM capacity ranges. They are not the infeasible full Cartesian product.

The remote launcher runs ten independent jobs concurrently on each of GPUs
1--3. The queue manifest is written to
`/data2/yb/remote_experiments/osram_mosi_hparam_sweep_20260918_parallel/`.
After the 90-job screen, `verify_top.py` automatically expands the five best
completed configurations to seeds 66--70 under the same protocol.
