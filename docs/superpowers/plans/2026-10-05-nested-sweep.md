# Nested module ablations and sensitivity

Goal: execute ten authorized seed66 MOSI runs, 100 epochs, unchanged cfg84
random missing protocol and per-rate BEST Test-oracle. INTERNAL DIAGNOSTIC ONLY.

Architecture: retain old Nested and Flat intact. Register ten explicit method
IDs, dispatch to a separate configurable graph adapter. No OSRAM changes.

## Locked experiment matrix

Reference: dim64, depth3, eight real-head nodes per evidence, rooted1-hop,
root/distance embeddings, same-role/same-head/Local edges, all-layer summaries.
Reuse existing Flat and old Nested seed66, never launch reference again.

Internal ablations: plain whole-graph GIN; no root/distance embeddings;
no cross-role same-head edges; last-layer summary only.
Sensitivity: dim32/128; depth1/2; grouped evidence nodes1/4. Grouped nodes
concatenate contiguous actual heads (512/group width), then project; decode
back to the same512 forward coordinates. OSRAM remains eight heads.

- [x] Agent implements gcnet_missing_m3/nested_sweep.py and module tests,
  plus registry/factory routing only. Tests first: initial identity, grouped
  shapes, inactive safety, finite updates and old-path regression.
- [x] Leader implements experiment runner/dispatcher, extending existing
  sealed-source runner choices with fixed block-ID-only config deltas.
  Tests first: ten unique configurations, unchanged protocol, mask verification.
- [x] Read-only spec and quality review, run focused CPU tests on biggpu s0.
- [x] Commit and push scoped files, seal tracked source with hashes. Check
  GPU0/1/6 UUID/free memory/disk. Start persistent dispatcher with bounded
  concurrent admission, no GPU4, no batch/optimizer alterations.
- [x] Record actual running/pending states and PIDs; all runs retain eight
  BEST checkpoints, predictions and recoverable last_training.pt. Summarize
  every completed/failed/pending configuration without hiding negative results.

No automatic multi-seed expansion or configuration combinations. Test-oracle
search is adaptive internal screening, not validation-selected paper evidence.
