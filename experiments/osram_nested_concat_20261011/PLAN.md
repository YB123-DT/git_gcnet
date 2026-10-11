# Nested pre-decoder concat: approved design and execution plan

INTERNAL DIAGNOSTIC ONLY. User approved the raw-evidence plus graph representation concat design and explicitly requested150 epochs. Existing frozen rooted probes did not train this joint readout.

MOSI seed66 only, two runs: nested_rooted_concat and nested_concat_capacity_control. Same server biggpu, physical GPU1 up to two concurrent jobs; exclude badGPU4 and drain-onlyGPU7. Existing Flat66 lower-LR150 is reused, not retrained.

Original observed encoder, causal Memory/query/write, Local path/skip and scalar regression head remain unchanged. Graph tokens: Local1, Base8, each active Gap8; original NestedGNN depth3/node64/rooted algorithm. Z uses the64d core.readout output immediately BEFORE old delta decoders, not the internal rooted pool tensor. Mean the eight head outputs separately for each memory role, retain Local64, concatenate five fixed64d roles. Inactive Gap and padding are safely zeroed before/after biased operations. Do not instantiate unused delta decoders.

Concat original4352d evidence and Z320 -> LayerNorm4672 -> Linear4672→1600 -> GELU -> Dropout -> Linear1600→1600; add unmodified LocalSkip, emotion_norm, original head. Preserve the existing zero initialization of final adapter, not zero-init an extra residual branch. Control: no graph, original4352 input with hidden width nearest to matching total trainable parameter count. Record actual counts and mismatch rather than claim exact equality. No auxiliary loss, mask change, completion, Gate or dual view.

Training100 epochs constantLR.001 then50 constantLR.0001 with full optimizer/model/RNG/BEST restoration. Per-rate BEST selection and original evaluation mask unchanged, original MOSI task loss. Scores remain internal diagnostic, no formal paper claim. Stage100 and150 saved separately with full training states,8BEST and predictions. No further seeds or search automatically.

## Implementation and verification

- Module owner: new nested_concat.py, minimal osram.py branch, tests/test_nested_concat.py. Existing disabled path unchanged; fork_rng isolates new initialization.
- Runner owner: register optional names, create this experiment's prepare/dispatch wrapper around verified complete150 train implementation. Save immutable source hashes, dataset hashes, effective config, environment and GPU UUID.
- Write failing tests, implement, verify shape4672/1600, five role slots, inactiveNaN/padding safety, baseline RNG/path, graph gradients and actual updates after zero-init warm start. Check control parameter mismatch.
- Compile/diff-check; commit and push only scoped files. Deploy immutable tracked snapshot; verify CPU tests on server and free memory/disk; launch persistent two-task controller with detached children; verify real training logs/PIDs and at least one epoch.
- Record running status, not fabricated results. Once complete report every rate, mean8/high, counts, final per-rate selected epochs and both100/150 scores.
