# Conditional second Memory read: free versus original-query span

INTERNAL TEST-ORACLE SCREEN

This directory contains four finite, from-scratch jobs: MOSI seed66 free/span
and MOSEI seed66 free/span. It does not train V3, TASR, a source tracker, a
new writer, a second classifier, or an auxiliary loss. Existing same-protocol
Flat66 references and their exact test masks are reused, not retrained.

## Question and computational path

For each original forward causal step, reuse the computed Local, original
forward Base, current availability, and the decayed pre-write Memory state:

`[Local256; Base512; availability3] -> LN771 -> Linear128 -> GELU -> Linear512`

Reshape to eight 64-dimensional queries and L2-normalize each head. The free
mode reads the query directly. The span mode first projects onto the span of
the actual original Base query plus currently active residualized Gap queries,
then normalizes and reads. Both modes use the same rank tolerances:
`rtol=1e-5, atol=1e-7`. Inactive Gap queries must never enter that span.

One extra `M_minus @ query` read is mapped through the same zero-initialized
`Linear512->512` and added only to forward Base. The backward half remains
zero. Original Gap, Local Skip, Flat1600, task head, block write and mask
sampling remain unchanged. A zero initialization must reproduce the fresh
original Flat prediction exactly; this is not pretrained initialization.

The span projection uses `torch.linalg.pinv` with the explicit thresholds and
is differentiable with respect to the original active queries as well as the
new query. No stop-gradient is added to the original queries. Only numerical
rank/out-of-span diagnostics are detached. Exact repeated, deficient and empty
query cases need finite-gradient tests before training. The two modes are an
architectural comparison, not a strict causal information test; their
backbones can evolve differently during joint training.

## Fixed budgets

| Dataset | Seed | Epochs | Learning rate |
|---|---:|---:|---|
| MOSI | 66 | 150 | epochs1-100:3e-4; epochs101-150:3e-5 |
| MOSEI | 66 | 60 | 1e-3 throughout |

Both use the existing s0 environment, official existing splits, batch32, Adam,
weight decay1e-5, original dropout, MSE, cyclic rates0.0-0.7 and original
`gcnet-random-mask-v1`. Selection is per-rate TEST-best weighted F1, strict
improvement with earliest ties. An eight-rate average can use eight different
epochs; it is not one checkpoint's deployment performance.

## Run once from an immutable source snapshot

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_conditional_reread_20261023.run \
  --source-commit <FULL_SOURCE_SHA> \
  --output <NEW_INDEPENDENT_REMOTE_OUTPUT> \
  --gpu-uuids <HEALTHY_GPU_UUID_1> <HEALTHY_GPU_UUID_2> --per-gpu 2
```

Run the controller under tmux on the established `ssh biggpu` server. Bind via
real UUIDs; physical GPU4 and its known failed UUID are rejected. The
controller requires >=64GiB available host RAM and >=50GiB free output disk,
does one real-data smoke per GPU at a time, then admits formal jobs using
measured peaks, live memory and current peer reservations. It never signals
other processes, shrinks batch size, alters the protocol, or expands the
four-job budget. `--resume` explicitly reattaches live workers or resumes
compatible full training states; it is not a weights-only continuation.

Each smoke performs at least four real optimizer updates. Parameter hooks
validate finite, nonzero gradients in the conditional query network and output
projection after the original/new zero exits have opened. The checks do not
use a stack of source tensors that was absent from the real computation graph.

## Auditable artifacts

- `QUEUE.json`, per-worker logs and `SUMMARY.json`: planned/running/failed/
  completed states, real PID/GPU UUID, time and measured peak memory.
- `config.json`, identity/source hashes/data manifest, initial parameter and
  RNG comparisons, paired initialization identity, exact test-mask hashes.
- `fixed_masks.pt`: exact masks matching the accepted Flat reference, labels,
  utterance metadata and saved evaluation batches.
- Eight BEST checkpoints with matching saved predictions/masks; `last.pt`
  includes model, optimizer, RNG, epoch, history and selected-point metadata.
- Real-data smoke verifies complete model/optimizer/RNG/output restoration.
- `history.json`: training loss, mask trace, original reference update/LR
  parity, selected-rate metrics, reread diagnostics and gradient coverage.
- `REPORT.json`, `PER_RATE.csv`, `RESULT.md`: completed-only per-rate and
  eight-rate/high-missing ACC/W-F1/MSE. Pair differences are **free minus span**
  within the same dataset; partial jobs and smoke scores are never ranked.

Parameter count, finite smoke results, actual runtime and any scientific claim
must come from executed artifacts, not this specification. Free not improving
over span rejects the new-address-access explanation for this implementation.
If both improve equally, a benefit cannot be attributed to access outside the
original query span. No automatic extra seeds, HPO, losses or query rounds.
