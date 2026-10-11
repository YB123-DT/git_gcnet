# Base-first, Gap-correction Fusion

INTERNAL DIAGNOSTIC ONLY — per-rate BEST Test-oracle.

Implementation/verification complete; formal performance pending. Approved budget:
MOSI seed66 only,150epochs. No sweep, other seeds or MC-FWC restart.

## Exact computation

`u = original_local_skip(L) + F_B([L,B])`

`delta = F_G([condition_LN(u), masked_Gap_A, masked_Gap_T, masked_Gap_V])`

`hidden = original_emotion_norm(u + delta)` then original task head.

F_B: LN1280 → Linear1040 → GELU → Dropout.5 → Linear1600.
F_G: LN4672 → Linear1040 → GELU → Dropout.5 → Linear1600.
The condition LN on u is separate from F_G's input LN. Both final Linear layers
are zero-initialized; F_B follows old adapter initialization, F_G begins with no
correction. Existing Local Skip and final normalization retain their original
parameters and initialization. Initial predictions consequently match the original
zero-adapter starting point, but the learned architecture is different.

Base/Gap keep their original1024-dimensional interface (forward512/zero512).
Gap slots are fixed and inactive slots are safely zeroed with torch.where.
No prior valid utterance or no active Gap means delta is hard-zeroed AFTER its
output bias; padding hidden is zero. Head bias is unchanged and loss uses umask.
One-stage joint training, only original task MSE, no extra losses/heads/Gates,
JEPA, completion, paired views or synthetic memory. CHM/read-before-write,
queries, original availability schedule and simultaneous block writes unchanged.

The new option is `--osram-base-first-gap` (default false), with Flat fusion.
Enabled mode removes the old adapter parameters rather than retaining an unused
large branch. Old checkpoints remain strict-compatible with default-off mode;
enabled architecture is trained from scratch, not loaded as an old checkpoint.

## Capacity

| Parameter count | Original | New |
|---|---:|---:|
| Adapter / two fusion MLPs including norms |9,535,104|9,538,464|
| Whole model |13,509,793|13,513,153|

Width1040 is the nearest integer budget match using a common width in both MLPs,
not a tuned value. Difference+3,360; not exactly parameter-matched. An enabled
forward has two MLP dropout operations rather than one; default-off RNG is
unchanged, but post-initialization training RNG trajectories are not claimed equal.

## Verification

29 targeted remote tests passed, including default legacy source/state/output/RNG,
unchanged write states/shared parameters, zero-init, hard mask after trained bias,
NaN inactive slots, warmed finite nonzero module gradients, parameter budget,
incompatible config rejection, original OSRAM and recovery-runner regression.
One upstream PyG deprecation warning. The initial remote legacy comparison could
not run git in an archive; supplied exact baseline source7d4bb0f as fixture and
reran successfully, without changing model arithmetic.

Real MOSI32-conversation843-valid-utterance GPU batch:3 Adam updates completed,
finite losses/gradients, peak allocated1,400,712,192 bytes. Losses3.043/202.746/
152.869 are initial engineering smoke values, NOT convergence/performance evidence.
See SMOKE.json. No claim that the new method improves sentiment performance yet.

## Training protocol / artifacts

Reuse original seed66 cfg84 config and original trainer. Only architecture flag
changes during first100epochs. Then reuse existing complete150 full-state
model/optimizer/RNG resume for50 more epochs atLR.0001 (first100 LR.001).
No validation selection: each epoch evaluates test rates0.0–0.7, separately retains
their best W-F1 checkpoints, exactly as the user requested. Report eight-rate and
high(.5/.6/.7) means, selected epochs and per-rate results. This is exploratory
test-selection, not one-checkpoint deployment or held-out generalization evidence.

Reference: original matched Flat seed66 lower-LR150, mean81.068095/high76.352251;
historical reference reused, not retrained. Both old and new use per-rate selection.
Features/labels and snapshot files are hashed in MANIFEST.json. GPU1 UUID verified;
GPU4 prohibited and GPU7 receives no new jobs. Training saves stage100/stage150,
full last_training.pt, eight best_miss checkpoints and predictions, history,
learning-rate trace and PROVENANCE.json. SUMMARY.json is written only after150
and original mask-hash checks complete. Existing cancelled experiments unaffected.

Entry: `python -m experiments.osram_base_first_gap_20261011.run --root ROOT --prepare --commit COMMIT`,
then from sealed snapshot with GPU1 UUID set,
`python -u -m experiments.osram_base_first_gap_20261011.run --root ROOT --train`.
Exact process/source/output details will be recorded in LAUNCH.json.

Files: new base_first_gap.py; local hooks/config/CLI in osram.py/model.py/train_gcnet.py;
test_base_first_gap.py and test_base_first_gap_runner.py; isolated experiment runner.
No new dependencies or unrelated refactoring. Validation establishes implementation
semantics, not that a Base-first ordering is better than simultaneous fusion.
