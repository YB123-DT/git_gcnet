# MOSI V3 direct-read experiment

User requests one MOSI variant without external read residual addition, passed
directly into Flat. This is a new from-scratch experiment, not checkpoint surgery.

- Keep Flat512, V3 width128/PairMLP256/four attention heads, output1600.
- Replace only enabled Base/Gap forward evidence with V3 decoders; no original
  read add-back. Internal V3 message/attention residuals remain part of V3.
- Keep original Local/Local Skip/Flat/task head, all CHM read/write/query math.
- Preserve original path for ATV/no history/inactive Gap/padding and read ablations.
- Direct decoders use their ordinary random initialization, not zero. Residual
  default retains exact old zero-initialization and RNG behavior. Thus this run
  compares direct+random-init against residual+zero-init, not addition alone.
- Seed66 only,100 epochs,3e-4 constant,batch32,Adam,WD1e-5,MSE,s0 environment,
  gcnet-random-mask-v1,cyclic0..0.7,exact original test masks.
- 100-epoch budget follows the user's earlier accepted reduction for this
  candidate (all24 three-seed residual best epochs<=92). Compare historical
  original Flat/Flat512/residual using their first100 epoch histories only.
- Internal per-rate Test-best W-F1; ACC/MSE from the same evaluation; not a single
  deployment checkpoint or independent generalization result.

## Execution

1. Write/run failing core and worker tests; implement opt-in mode and forwarding.
2. Run targeted legacy/RNG/masking/causal/gradient/reload tests and compile checks.
3. Commit immutable source, push only github/feature/ecerc-features, archive to
   independent biggpu directory. Existing running source snapshots stay untouched.
4. On an available healthy GPU (never4), run one two-epoch/four-update real MOSI
   smoke, validate all V3 gradient groups/single scan/Adam+RNG restoration.
5. If smoke passes and measured resources allow, controller automatically starts
   one fresh100-epoch run. Preserve8BEST+same-call predictions/masks+complete last.
6. Completed-only summary compares first100-epoch references with matched masks,
   records all initialization and source differences, and does not expand scope.
