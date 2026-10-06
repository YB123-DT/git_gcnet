# Old Nested implementation versus proposed narrative

Reviewed2026-10-06 at a04692e, method nested_gnn_rooted_evidence.
Scope: code review only, no model edits, new training or inference sweep.
Independent native reviewer cross-checked the leader's findings. No blocking
implementation defect or cross-utterance/cross-batch interaction found within
the inspected path. This does not establish task performance or full paper
reproduction. INTERNAL DIAGNOSTIC ONLY.

## Exact forward path

ObservedSetEncoder -> original causal OSRAM scan -> original local path
L=node+local_path(node), Base/active Gap reads -> typed head tokens -> rooted
GIN -> zero-initialized decoder bridges -> adapted Flat adapter inputs.
Original Local Skip receives ORIGINAL L, not graph-adapted L.

Let B/G below denote the safely masked classification reads. The old block
produces L'=L+deltaL, B'=B+deltaB, G'=G+deltaG, with modifications restricted
to forward512 of history reads. Final representation is:

```
h = emotion_norm(local_skip(L) + emotion_adapter([L', B', G'_A, G'_T, G'_V]))
```

It is NOT h=LN(original_flat_anchor+1600D_graph_residual). Original Flat
modules/head remain, but adapter INPUT VALUES change. First valid utterance
and rows with all forward reads zero bypass graph processing. Padding and
inactive Gap are excluded safely. Modules remain jointly trained.

## Grounded claims

| Claim | Code evidence | Permitted statement |
|---|---|---|
|Typed nodes|meaningful_blocks_common.py:16-46|Local256->64; each true memory head64->64 with per-head Linear shared across evidence roles, plus role/head embeddings and LN|
|Active nodes|meaningful_input.py:66-78|Base plus missing-modality Gap; current availability determines the active graph|
|Fixed graph|meaningful_new40_structure.py:20-27|Same-role edges, cross-role same-head edges, Local-to-all; not learned topology|
|Nested encoding|meaningful_new40_structure.py:339-360|One induced1-hop subgraph per node; shared3layer GIN-style sum/MLP/LN|
|Root designation|meaningful_new40_structure.py:345-357|Binary input root/distance markers; mean of ALL subgraph nodes per layer|
|Output decoding|meaningful_input_new40.py:26-43|Residuals for adapter Local and original-shaped Base/Gap slots; zero bridges at initialization only|
|Flat integration|osram.py:1748-1766|Original Local Skip preserved; adapted Local/Base/Gap feed adapter|
|No new temporal memory|osram.py:1593,1749-1766|Insertion after causal scan, no output writeback or second query|

## Narrative qualifications (not hidden implementation fixes)

1. Current-read nodes are not historical utterance nodes. With at least one
   observed modality, active node counts are9/17/25, not33. The33 fixed slots
   are capacity before masking.
2. Local connects to all nodes: its1-hop induced subgraph is the whole active
   graph. In ATV, only Base exists and the9-node graph is complete; all roots
   have the same vertex set, differentiated by root markers. Do not claim
   every root produces a different small neighborhood.
3. The old output includes each node's own rooted summary AND global mean
   rooted summary. Do not describe it as exclusively local relational output.
4. Original old pooling is not center-hidden extraction nor separate
   root/neighbor pooling; those belong to the separate negative root-aware run.
5. The distance marker is graph distance0/1, NOT elapsed conversation time.
   For1-hop induced subgraphs, root and distance embeddings provide redundant
   binary root/nonroot information; they are not two independent semantic axes.
6. Availability is mask/topology support, not a separately learned availability
   vector input to this Nested core. Type/head identities do enter explicitly.
7. Memory equations and observed-only writes remain unchanged, but joint task
   gradients may change upstream/Memory weights. Separately trained Flat and
   Nested checkpoints do NOT have guaranteed identical memory trajectories.
8. No direct gold-label input, reliability estimator, missing-feature recovery,
   emotion-shift detector or guaranteed conflict-resolution capability exists.

## Verification and performance boundary

Fresh16 CPU tests passed12.82s on biggpu s0 sealedsource0b9b36b:
tests/test_nested_sweep.py and tests/test_nested_rootaware.py. They cover
old seeded RNG/parameters/output regression, zero initialization, masks,
finite gradients/parameter updates, first utterance and upper-half preservation.
Existing22-test runner/integration verification is separately recorded in RESULT.
No GPU smoke/new training was launched by this review.

Old Nested three-seed mean80.489/high75.252 versus Flat80.559/high75.594.
These are Test-oracle internal diagnostics, not evidence of overall improvement.
The current evidence permits describing structured evidence interaction as a
design motivation; it does not permit claiming demonstrated bottleneck repair.
