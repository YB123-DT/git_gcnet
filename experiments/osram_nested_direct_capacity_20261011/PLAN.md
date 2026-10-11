# Direct random Nested capacity screening

INTERNAL DIAGNOSTIC ONLY. User authorizes several larger direct random Nested
versions; first keep Flat hidden1600/output1600 unchanged. No additional seeds.

| Method | Node dim | Internal MLP width | Depth | External addition | Decoder init |
|---|---:|---:|---:|---|---|
| Reference direct random |64|64|3|none|normal Linear|
| nested_direct_random_mlp256 |64|256|3|none|normal Linear|
| nested_direct_random_mlp512 |64|512|3|none|normal Linear|
| nested_direct_random_dim128 |128|128|3|none|normal Linear|
| nested_direct_random_dim256 |256|256|3|none|normal Linear|

Measured module parameters: reference159235; MLP256332227; MLP512562883;
dim128432387; dim2561322755. These are module counts, not whole model counts.

Use existing NestedSweepGNN with original rooted topology/mean pooling,
markers and eight head nodes; same Tanh, normalization, decoder interfaces.
Original Local Skip and Flat/head retained. OSRAM 8x64/read/write unchanged.
Width changes internal RNG draw consumption; no claim of identical initialization.
Original direct random reference W-F1 mean8/high 80.176918/74.800208;
Flat81.068095/76.352251; residual Nested80.992147/76.077229.

Protocol: MOSI seed66 from scratch,100epochs, original random cyclic0..7,
per-rate Test-oracle BEST, Adam.001 constant, batch32, same task loss and data.
No Gate/completion/JEPA/newloss/lr changes. Independent sealed code/output paths.
biggpu physicalGPU7 only, GPU4 forbidden, two concurrent admissions with measured
memory margin. Four approved runs only, no automatic further sweeps.

Implementation plan (direct execution authorized by user):
- Write/run four registration/interface/no-bypass/finite-gradient tests RED.
- Extend existing NESTED_SWEEP options with opt-in residual/decoder settings;
  defaults preserve all existing residual variants.
- Run tests GREEN and record actual module parameter counts.
- Commit/push, archive tracked source and SHA256 manifest, reuse dispatcher.
- Verify live PID, effective RAW_CONFIG and first training progress; final results
  require100epochs, all checkpoints/predictions and outputs_verified provenance.
