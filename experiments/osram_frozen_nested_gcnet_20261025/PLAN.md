# MOSI66 original Nested on protected frozen Flat parents

Approved by the user after reviewing the completed NoArea serial/parallel run.
Reuse the existing feature worktree; do not change old modules or experiments.

## Locked scope

- Eight new runs only: seed66, MOSI, each accepted own-rate Flat checkpoint at
  rates0.0–0.7. Parent weights, masks, data and splits are the same as the completed
  `osram_recent_noarea_serial_20261025` stage2 comparison.
- Original `nested_gnn_rooted_evidence`: dim64, three GIN layers, eight real
  memory heads; not a sweep, new topology, direct replacement or root-aware variant.
- Correct Local/Base/Gap at the original Flat input; keep raw Local Skip,
  CHM joint write/read/query and original task head untouched.
- Freeze all parent parameters, buffers and Dropout. Train only the existing
  Nested adapter with Adam, batch32, WD1e-5, original MSE including neutral labels.
- 150epochs; exact LR expression3e-4 for epochs1–100 and3e-4/10 for101–150.
- Cyclic GCNet-random-mask-v1 training; each run evaluates only its own fixed
  test rate. Earliest strict Test-best W-F1, ACC from the same saved predictions.
- INTERNAL TEST-ORACLE SCREEN. Eight selected parents/blocks are not one deployed
  checkpoint. No extra seeds, methods or tuning.

## Execution plan and acceptance evidence

1. Implement a small external frozen wrapper, reusing MeaningfulInputAdapter.
   Test zero-init identity, branch-off, unchanged Local Skip, active slots,
   frozen input-gradient flow, one live scan, and strict state restoration.
2. Reuse stage2 sampling, optimizer, metrics, trace and recovery utilities from
   the completed NoArea run. Require exact own-parent and test-mask replay;
   compare each training mask/rate/update trace with its completed reference.
3. Test finite eight-job queue, healthy GPU UUID binding and artifact validation.
   Run one real MOSI two-epoch one-batch GPU smoke before formal admission.
4. Save an immutable source snapshot; use healthy physical GPU0/1/2 with a
   measured-memory queue, maximum four of this task's processes per GPU.
   Never use GPU4, kill other tasks or overwrite protected artifacts.
5. Run all eight jobs, restore every selected block and recompute all saved
   predictions. Report per-rate and eight-rate/high-missing ACC/W-F1, best epoch,
   branch-on/off corrections/harms, provenance, cost and failures.
6. Commit and push only scoped code/tests and small launch/result records to
   github feature/ecerc-features. Preserve all weights, arrays and old records.

Model/tests may be implemented by one bounded native subagent; the leader owns
runner/scheduler integration, remote checks, launch and final verification.
