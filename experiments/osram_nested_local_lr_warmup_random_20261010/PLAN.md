# Nested local lower-LR + warmup + random decoder initialization

INTERNAL DIAGNOSTIC ONLY. One MOSI seed66,100epochs, from scratch.
User explicitly requests lower local LR, warmup, no zero initialization.
This is NOT Flat-inside-GNN redesign; original residual-input Nested architecture
remains intact. Keep OSRAM/Flat/task loss/head/missing masks/evaluation unchanged.

Nested module means ALL osram.meaningful_block parameters (tokenizer, graph core,
residual decoders), not OSRAM memory parameters. Independent Adam group target
LR5e-4; linear5epoch warmup:1/2/3/4/5e-4, then5e-4 constant to100. All other
groups keep1e-3 constant. Retain original GLOBAL clipping norm1.0.
Replace zero-initialized local/head residual decoders by default nn.Linear random
initialization, residual=True. Do NOT use nested_gnn_direct_random_evidence:
that historical variant disables the residual and changes the experiment.
Initialization no longer equals Flat; record initial decoder norms/nonzero state.
No claim this recipe guarantees improvement or isolates the effects of3 changes.

Reuse sealed source_ad211c0, reference completed Flat seed66 original100 config,
same data manifest. Runtime policy is isolated in this experiment; no historical
source edits or running-experiment source mutation. Save wrapper/policy hashes,
effective config, actual per-groupLR each epoch,8 BEST and full recovery state.

Implementation files: variant.py (random factory/group split/warmup), run.py
(sealed-source execution/provenance), tests/test_nested_local_lr_warmup_random.py.
- [ ] Write3 minimal regression tests, run RED, implement policy, run GREEN.
- [ ] Run original monitored-mask/interface regression coverage without new GPU smoke.
- [ ] Commit/push; copy immutable runner/policy to remote; check GPU7 resources.
- [ ] Launch one persistent job; verify random decoder norms, disjoint4 optimizer
      groups, actual epoch1 Nested1e-4 and other groups1e-3; log provenance.
- [ ] At completion compare original Flat/Nested seed66 under same100epoch budget,
      report per-rate8-rate/high means. Do not automatically expand seeds/search.

Server biggpu physicalGPU7 UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e;
GPU4 forbidden. Prelaunch freeGPU7 19685MiB; /data1 free76GiB. One additional
~3GB-class Nested run fits without killing other processes or changing batch32.
