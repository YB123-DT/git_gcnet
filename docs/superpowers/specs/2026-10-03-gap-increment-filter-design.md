# Gap increment scalar filter

User approved the preceding Gap-increment placement with “那就加一个看看行”.
This records the fixed minimal instantiation; no architecture/hyperparameter search.

Original cfg84 no-JEPA Flat remains default. Add --osram-gap-increment-filter,
default false, only compatible with original causal Flat/shared head, no other
gates, Relation, dual supervision, post-GRN, completion or paired/persistent views.

Compute one Local and Memory trajectory. Original adapter A and skip S are shared:
uF=S(L)+A([L,B,maskedG]); uB=S(L)+A([L,B,0]); dG=uF-uB.
For numerical identity use equivalent dG=adapterF-adapterB and
hidden=original_LN(uF+(g-1)*dG). Preserve original full adapter evaluation order.

One scalar per valid utterance:
g=1+tanh(Linear128to1(GELU(Linear(2D+3,128)([LN(uB),LN(dG),availability])))).
D is actual output_dim, not hardcoded. Gate last weight/bias zero-init. Other
initialization normal inside fork_rng so existing weights and downstream RNG
are unchanged. Gate has no dropout and no auxiliary loss. Range (0,2) allows
attenuation/amplification; g=1 is exact original Full at initialization.

Two Adapter calls must share the exact dropout draw. Capture RNG immediately
before original full adapter; replay base adapter under fork_rng restoring that
pre-call CPU/device state, so global post-forward RNG advances exactly once.
No detached uB/dG or frozen parameters. No second Memory/query or task head.
Safe torch.where masking for padding and inactive Gap. No-gap positions have
dG=0 with shared dropout; padding output strictly zero.

Diagnostics: valid gate mean, fraction below0.1/above1.9, dG norm, modulation
norm ||(g-1)dG||, modulation/full-anchor norm ratio. Not reliability estimates.

Correctness: flagoff exact outputs/state/RNG; identity-on eval/train and common
gradients at zero-init; g0/g1 endpoint algebra; shared dropout; inactive NaNs
cannot leak; paddingzero; one scan; finite updates and filter learns; checkpoints
round-trip; default old checkpoint loading preserved. Run actualcfg84 CUDA checks.

Experiment: original raw Flat config, seed66,100 epochs, all original eight-rate
cyclic masks/optimizer/loss/batch/per-rate BEST Test-oracle. Only new flag differs.
From scratch, one-stage joint training, full output only. No additional seeds.
Run on biggpu healthy GPU5 in immutable snapshot; retain all BEST files remotely.
Report per-rate/overall/high WF1 plus gate diagnostics and parameter count.
INTERNAL DIAGNOSTIC ONLY; no validation-selected paper claim.
