# Scalar MemoryShiftFilter depth/width experiment

User requested deeper/wider filter MLPs and concurrent execution. Four fixed
variants, seed66,100epochs, original cyclic random-missing no-JEPA training:

|Name|MLP (GELU after each hidden layer)|
|---|---|
|D2-W128|768→128→128→1|
|D2-W256|768→256→256→1|
|D3-W128|768→128→128→128→1|
|D3-W256|768→256→256→256→1|

Depth counts hidden layers, not all Linear modules. Output remains one sigmoid
scalar per evidence, multiplied by shift. Relation space/type width128, active
slot mean, zero-initialized residual projector, original trainable Flat anchor,
Memory read/write/query and task head remain unchanged. No vector gate, new loss,
dual-view training, completion, JEPA, or persistent masks. Default depth1/width128
preserves the existing module/checkpoint layout; Flat remains the default readout.

Reuse existing Flat and old scalar MemoryShiftFilter seed66 results; no reference
retraining. Existing old scalar seed66 eight-rate80.4968/high75.8126; Flat
81.0681/76.3523. Both references are per-rate Test-oracle internal diagnostics.
This is an exploratory capacity comparison, not a validated improvement claim.

Verification before training: default state-key/RNG compatibility, exact initial
Flat equivalence, masks/padding, scalar gate output shape for all variants,
finite backward gradients and nonfrozen original paths; four configurations must
differ from old scalar module only by explicit MLP width/depth. Separate outputs
retain effective configs, source hashes, provenance, logs and8best checkpoints.

Server biggpu. HostGPU4 forbidden. GPU6 currently full with unrelated jobs;
GPU5 initially has about16GB free. Run four independent processes onGPU5 if
measured capacity permits, stagger starts briefly but do not serialize training.
Do not kill others' jobs or change batch size to force concurrency. Coordinator
continues checking exit status in a persistent server-side process.
