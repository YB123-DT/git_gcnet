# WSC-OSRAM: Write-State Completion

Approved independent model; implementation/verification only, no full training.
ObservedSetEncoder and causal eta=.6 Flat OSRAM remain. Current utterance reads
old memory; predictor uses node, mean real observed latents, Base and target Gap
to predict missing target K/V. Keys normalized per head. Real slots kept exact;
one simultaneous block ridge write with all valid slots. Natural availability
never changes. Predicted writes can affect future reads but not current reads.

Shared predictor with target embedding3x16, LayerNorm→Linear→GELU→Linear.
Input dimension2*latent_dim+2*(H*dv)+16; outputH*(dk+dv). Integer hidden width
minimizes parameter difference from old complete-state head budget. Default
reference700-hidden/256-latent budget250872. No trainable gates/new losses/sweeps.
The new supervision replaces old JEPA: mean SmoothL1 on normalized K plus mean
SmoothL1 on raw V, averaged .5/.5 over truly missing valid slots; total task+.1aux.

Teacher: EMA observed encoder plus copies of OSRAM K/V projectors, latent/node
norms, availability/speaker embeddings. Teacher key combines full target modality
latent with observed node, original availability/speaker. Teacher value uses full
target modality latent. No Teacher memory, classifier, or future/label input.
Initialize copies exactly; EMA after optimizer. Save EMA step as module buffer.

Missing writes use unchanged per-target beta, alpha/ridge/step unchanged.
No special prediction confidence, protected write, completion read node, second
scan, or new readout. Predictions are retained in inference. Full raw targets
exist only in explicit training supervision call, never model.forward.

## Implementation plan / acceptance

- Test first: default exact outputs/state/RNG; one write per active time;
  observed K/V exact; natural mask unchanged; current reads unchanged by current
  write; future reads may change; conversation reset/padding/causality.
- Add write_state.py predictor/EMA target/loss and one optional _scan callback.
  Keep default callback absent and old arithmetic unchanged.
- Integrate model opt-in write_state_completion and write-state trainer objective.
- Verify finite gradients from future emotion into predictor after nonzero
  readout initialization; auxiliary gradient; no Teacher gradients/forward calls
  during test; EMA exact, strict loading, nearest-width parameter count.
- Existing OSRAM/causal/retention/state training tests and one real batch check.
- Document parameter count and save ready-to-run MOSI config; do not launch.

Risk: prediction error enters persistent memory and may compound. Equal total
MLP capacity does not establish equal function or compute; frozen Teacher and
legacy retained modules must be counted separately. No improvement claims.

## Implemented interface

`MissingM3GraphModel(..., write_state_completion=True)` enables WSC. Defaults
remain false. Trainer `--training-objective write-state` reconstructs this flag
from saved config, retains predictor in train AND inference, but calls target
supervision only during training. It cannot combine with complete-state/B2 or
legacy classification completion. `write_state_loss(full, availability, qmask,
umask)` uses cached predictions from the latest training forward.

Main implementation: gcnet_missing_m3/write_state.py. Integration: model.py,
osram.py optional post-read write callback, train_gcnet.py objective switch.
No modification of beta, alpha, ridge solve, Local/Base/Gap definition or Flat.
The teacher target uses EMA parameters with the original availability; missing
raw features are used only for modality target K/V, never the student forward.

## Measured size and smoke

| Item | Parameters |
|---|---:|
| Reference complete-state online head |250872|
| WSC online write predictor (width160) |251120|
| Difference |+248 (+0.0989%)|
| WSC frozen target generator |1916160|
| WSC all trainable |4853641|
| WSC all registered (includes unused legacy frozen modules) |9348497|

The online budget is matched, NOT the registered or frozen Teacher count.
Nearest integer width computed from actual input/output dimensions, not hardcoded.

One actual MOSI batch on GPU5:32conversations,843utterances,1142missing slots.
Two updates; finite gradients. Task→predictor gradient is initially zero due to
the inherited zero-initialized emotion adapter, then nonzero on update2.
Update2 task gradient L1=7649.70. Teacher gradients absent; two EMA updates saved
as `write_state.ema_updates`; eval calls no Teacher. Smoke total loss rises
2.8547→40.3348: this demonstrates execution/gradient connectivity, not convergence.
No full training or score results exist for this model.

Ready config: config.seed66.json (cyclic MOSI,100epochs,per-rate Test-oracle).
It is not launched. Main CLI accepts the saved config's fields via existing flags.
The raw smoke evidence is SMOKE.json. Old checkpoint/default compatibility is
tested separately; new WSC checkpoints restore strictly with the new mode.
Existing checkpoint format is selection/model-state storage, not full optimizer
resume. Registered EMA step makes target update count recoverable for this mode.

Verification:62 distinct related tests passed across component/trainer/complete-state/
OSRAM/write-step/retention/B2 suites. Historical Git-based tests required injecting
their existing base64 source inputs because remote code mirror has no .git;
those24 tests then passed, including all three initially blocked checks. Only
the existing PyG deprecation warning remains. Independent read-only review found
no blocking causal/autograd/target-isolation issues. git diff --check clean.
