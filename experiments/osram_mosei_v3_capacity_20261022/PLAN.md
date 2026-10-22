# MOSEI smaller-Flat / wider-V3 finite implementation plan

User authorization: lower Flat capacity and increase V3 capacity on MOSEI.
Two seed66 runs only: Flat1024 control and Flat1024+V3(dim256,pair-hidden512).
Reuse completed Flat1600 / original V3 results read-only; do not retrain them.

Architecture: Flat intermediate hidden width1600->1024 via the existing
osram_adapter_hidden_dim option. Flat output1600, Local Skip and task head
remain unchanged. V3 keeps its hierarchy, four roles, three sources, masks,
zero-init four512d decoders and4 attention heads; only channel/PairMLP widths
increase64->256 and128->512. CHM8x64 and all original query/write math stay fixed.

- [ ] Add failing opt-in capacity/default-state compatibility tests, then expose
  V3 width64/pair-hidden128 defaults through config/model/CLI. Reject noninteger
  or nonpositive widths and widths not divisible by4.
- [ ] Test a bounded MOSEI config builder: only adapter width and optional V3
  switch/widths can differ from the accepted Flat config. Both arms are fresh
  joint end-to-end training, same model initialization outside added V3.
- [ ] Implement one dedicated worker reusing old MOSEI setup, train_epoch,
  original MSE, common exact test masks, persist_epoch/full last state and
  strict W-F1 Test-best selection. Require matching training mask hashes,
  optimizer steps and LR against the original60epoch trajectory.
- [ ] Run CPU regression/default parity plus three-real-batch GPU smokes for
  both arms. Verify initial V3 output equals small Flat, all V3 gradient groups
  open, one CHM scan per training batch, finite updates and exact checkpoint
  model/Adam/RNG/prediction reload. Smoke is not a performance result.
- [ ] Commit/push only related implementation/tests/docs; deploy one immutable
  Git snapshot to biggpu. Use archived s0 runtime, health-verified GPU1 UUID,
  neverGPU4. Measure two-job peaks before concurrent admission; preserve all
  other tasks. No batch reduction or protocol changes for fitting.
- [ ] Start exactly two60epoch runs in tmux; record PID/config/source/masks/
  parameters/peak memory and progress. Queue auto-reports per-rate ACC/W-F1/MSE,
  eight-rate/high means versus same-round small Flat and historical large Flat.

Fixed protocol: CMUMOSEI,seed66,60epochs,Adam,constant LR1e-3,WD1e-5,batch32,
same original dropout,emotion-only MSE,cyclic0.0..0.7,gcnet-random-mask-v1,
per-rate TEST-best W-F1,exact accepted Flat66 test-mask replay.
Label all scores INTERNAL TEST-ORACLE SCREEN, not independent generalization.
No additional seeds, HPO, modules, JEPA, gates, completion or dual views.
All new-Miss checkpoints remain retained, including smoke/recovery artifacts.
