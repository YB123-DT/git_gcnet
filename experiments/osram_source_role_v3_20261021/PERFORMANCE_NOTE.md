# Observed training cost and shared-GPU contention

2026-10-21; read-only inspection of existing queues, logs, processes and
`nvidia-smi`. No task was stopped, migrated or reconfigured.

- MOSI V3 completed150 epochs in2448.53seconds, approximately40.8minutes.
  Its history records2 optimizer batches per epoch; MOSEI records71.
- Each formal epoch also evaluates all8 missing rates and saves the full
  recovery state, plus newly improved per-rate selections. Therefore epoch
  cost is not training-only and parameter count is not a runtime estimate.
- Physical GPU1 utilization100%, memory32480/32768MiB. V3 MOSEI PID3638587
  occupied6426MiB; two processes outside the approved V3/Nested queues occupied
  3872MiB and22176MiB. The card was initially free at V3 launch; these
  additional processes appeared later. They were not modified.
- Physical GPU7 utilization99%; the3 Nested seeds shared it with another
  process outside this queue occupying4280MiB. GPU0's2 Flat controls also
  shared their card with a separate13764MiB process.
- All6 remaining experiment workers were present and running without a
  recorded error. Host available RAM was302GiB; inspection did not show host
  RAM exhaustion.

These observations establish additional workloads and shared-resource
contention, but do not quantify each process's share of compute or isolate
training/evaluation/I/O/operator timing. No profiler or new benchmark was run.
Earlier2.5–3hour remaining-time estimates used elapsed/completed-epoch
averages, not a measured recent-epoch window; they may be optimistic if
contention persists or grows. No completion deadline is guaranteed.
