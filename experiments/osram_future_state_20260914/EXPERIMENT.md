# Approved Future-State training launch

User approved launching after implementation/smoke commit `3f88cb3`.
The earlier RESULT.md and SMOKE.json describe pre-launch verification, not
training results. This document records the subsequent training phase.

- MOSI, seeds 66–70, one cyclic eight-rate model per seed, 100 epochs.
- Causal OSRAM .6, mean/Flat; all source config values inherited except objective
  and the already-approved per-rate Test-oracle selection metadata.
- No WSC, B2, old MMoE forward, extra loss, or baseline retraining.
- Each seed/rate independently selects its highest Test W-F1 epoch. A mean of
  selected scores is descriptive only, never a checkpoint selector.
- Five concurrent jobs on GPU 6, selected after checking current allocations.
  GPU 4 is excluded. Other users' processes are untouched.
- Remote root: `/data2/yb/remote_experiments/osram_future_state_20260914`.
- `QUEUE.json`, per-seed logs and `mosi/seed_N/PROVENANCE.json` record execution.
- Every run checks final mask hashes against its inherited reference.

INTERNAL TEST-ORACLE DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT.
