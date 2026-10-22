# MOSI Flat512 + V3-128 three-seed supplement

Add exactly two fresh joint runs,seeds67/68. Reuse the existing seed66 arm
`osram_mosi_v3_capacity_20261022/execution_v1/runs/f512_v128`;do not repeat it.
Flat hidden512/output1600,V3 width128/PairMLP256/four heads,total7,864,639.
No small-Flat67/68 extra training,HPO,other datasets or mechanism changes.

Keep MOSI150epochs,LR3e-4(1..100)/3e-5(101..150),batch32,Adam,WD1e-5,MSE,
archiveds0,official split,new gcnet-random-mask-v1,cyclic0..0.7,original CHM.
The user's earlier100epoch change was conditional;its all-best<=100 condition
was not met,so this supplement keeps the actual seed66 budget.

- [ ] Tests-first narrow seed67/68 entry point and default-preserving base-worker
  dependency injection. Each seed validates against its OWN completed original
  Flat history/initialization/training order/masks/LRs;test masks remainseed66.
- [ ] Implement persistent two-run queue on healthy idle biggpu GPU2,two real
  four-update smoke checks then both formal runs concurrently if memory permits.
- [ ] Report separate seeds and per-rate ACC/WF1/MSE against original Flat66/67/68.
  Only compute a three-seed mean/SD when all three candidates are complete.
  Compare unchanged model/math hashes across candidates;new entrypoint hashes
  can differ from the original seed66 immutable snapshot and must be recorded.
- [ ] Regression tests,commit immutable snapshot,deploy,real smoke/launch evidence,
  commit/push small records. Preserve all old/new weights and MOSEI jobs.

Internal per-rate TEST-best WF1 only,not single-checkpoint independent test
generalization. This is multi-seed verification of a seed66-screened candidate,
not a parameter-matched three-seed proof versus smaller Flat alone.
