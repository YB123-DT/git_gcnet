# Authorized two-stage MOSI run

Started 2026-09-14 approximately 14:04 UTC from implementation commit `010d568`.
User explicitly authorized launch after engineering verification.

Seeds: 66, 67, 68, 69, 70. Each seed runs Teacher first, then Student only if
Teacher exits successfully. The shell chain uses `teacher && student`; failures
do not silently proceed with missing/partial Teacher weights.

- Teacher: complete inputs, emotion-only, up to 100 epochs, validation selection,
  no Test evaluation. Export trained `observed_set.projectors.*`.
- Student: fresh baseline initialization, frozen exported Teacher, original
  contextual MMoE and JEPA, cyclic rates 0.0–0.7, 100 epochs.
- Student selection: independent best Test W-F1 epoch per seed × rate.
  INTERNAL DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT.
- Both stages retain inherited batch size 32, optimizer learning rate 0.001,
  causal OSRAM write step 0.6, Flat, mean fusion. No architecture changes.

Remote checkout: `/data2/yb/paper/GCNet_TPAMI_missing_m3_target_ple` on `biggpu`.
Python: `/data2/yb/reproduction_envs/s0/bin/python3.10`.
Output root: `/data2/yb/remote_experiments/osram_supervised_teacher_20260914`.
Logs: `seed_66.log` through `seed_70.log` under that root.
Each stage writes its own config, provenance, history, metrics and COST.json.

GPU placement: seeds 66/67 on GPU0; seeds 68/69/70 on GPU6.
GPU7 failed CUDA initialization for seed70 before training; its original failed
directory/log are preserved as `teacher/seed_70_cuda_init_failed` and
`seed_70_cuda_init_failed.log`. Seed70 was restarted from scratch on GPU6 using
its UUID. GPU4 was not used. No unrelated process was stopped.

`RESULT.md` and `SMOKE.json` describe earlier engineering verification only.
This launch is not evidence of a performance gain. Full training results remain
pending and must be extracted after completion, including Teacher integrity and
Student mask-hash checks.
