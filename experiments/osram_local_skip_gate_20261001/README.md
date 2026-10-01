# Local Skip-only scalar Gate

Approved model: `g=1+.2*tanh(G(Local,masked Base/Gap,availability))`, one scalar
per valid utterance. `hidden=LN(g*local_skip(Local)+Adapter([Local,masked C]))`.
Gate multiplies full Skip output including bias. Adapter inputs and Memory
read/write/query paths unchanged. Use original Flat, actual output_dim1600.

G reuses the established128-hidden ELU scalar network (HistoryInputGate network
only), with zero output-layer weight/bias: starts at1. Shared initialization
RNG preserved. Default-off flag `--osram-local-skip-gate`; mutually exclusive
with other gate/readout adaptations. Padding and inactive Gap safely masked.
Conditions use already-ablated emotion contexts, never bypass readout ablations.

One-stage joint training from scratch; all original modules trainable. No extra
Gate regularizer or loss, no JEPA, prediction completion, persistent mixture,
attention or vector gate. MOSI original regression MSE retained for training
and recording; assessment emphasizes W-F1, not an MSE-improvement requirement.

Run3seeds66/67/68,100epochs, original cfg84 cyclic missing0–.7, batch32,
Adam lr.001 weight decay1e-5. Original matched Flat references reused, not
retrained. Retain eight best checkpoints per seed. Selection is inherited
per-rate Test-oracle: internal diagnostic, not formal validation-selected score.

Remote biggpu hostGPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153, never GPU4.
Code and outputs under `/data1/yb/remote_experiments/osram_local_skip_gate_20261001`.
Coordinator staggers up to3children, requiring5GB free before each admission.
Record config/reference/source hashes, PID and logs; isolate running code.

Verification before launch: disabled/zero-init train/eval identity, RNG and
Memory contexts unchanged, complete Skip bias scaling, adapter unaffected,
inactive/padding safe, finite joint updates, CLI/config and diagnostics.
