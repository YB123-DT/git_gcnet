# MOSEI — four fixed Nested combinations

INTERNAL TEST-ORACLE SCREEN. User accepted the first FOUR proposals after the
seven-run cmerc screen. No fifth MLP256, extra Flat, HPO, seed or dataset.

| Method | Graph dim | Depth | Graph groups | MLP hidden | Nested parameters |
|---|---:|---:|---:|---:|---:|
|mosei_nested_d32_l3_g8_m128|32|3|8|128|109155|
|mosei_nested_d32_l3_g8_m64|32|3|8|64|80163|
|mosei_nested_d32_l2_g8_m128|32|2|8|128|96642|
|mosei_nested_d32_l3_g4_m128|32|3|4|128|108899|

MLP hidden controls GIN layers, pool and readout, not just the final layer.
Default Nested32 had MLP32. Graph groups change only tokenizer grouping; actual
CHM heads remain8, KV64, forward512/backward-zero512. Local256/Skip/Flat1600,
causal joint BlockWrite,read-before-write,write_step0.6 are unchanged. Original
Nested input residual and zero decoder remain. No new algorithm, auxiliary loss,
Gate,JEPA,V3,Router or paired views. One-stage end-to-end original MSE.

MOSEI seed66/fold1,60epochs,Adam constantLR1e-4,WD1e-5,batch32,original dropout,
cmerc-repro and accepted original feature/split manifests. GCNet random-mask-v1
cyclic0–0.7 training; existing accepted exact test masks and per-rate TEST-best
W-F1 selection; ACC/MSE from the same predictions. Retain eight BEST, exact
predictions/masks,history,last model/Adam/CPU+CUDA RNG. Existing UUID RNG repair
runs before initialization, exactly as the matched screen. No old weights loaded.

Reuse completed read-only Flat/Nested32/MLP128 controls in
`/data2/yb/remote_experiments/osram_nested_cmerc_tuning_20261023/execution_v1/runs/CMUMOSEI/fold_1/seed_66`.
Compare source files byte-for-byte against that screen's immutable source_v2;
the sole numerical-package difference allowed is these four registry entries.
The shared experiment worker changes ONLY CLI/method scope admission, preserving
the old seven-method METHODS tuple and all historical queue/report plans. Current
live IEMOCAP snapshots are not edited. Final report verifies original init,
config,split,mask bank,epoch mask traces,LR/update counts and artifact integrity.
Different source SHAs are disclosed, not treated as unconditionally equivalent.

biggpu physicalGPU2 UUID `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4` only, process
cuda:0. PhysicalGPU4 forbidden. GPU2 was idle before admission. Completed same
protocol MOSEI peaks ranged1779–2476MiB PyTorch allocated. Reserve6144MiB per
worker plus2048MiB global margin (including CUDA context overhead), at most4
workers, >=64GiB available RAM and25GiB disk. External allocations stay charged;
workers still loading CPU data reserve full memory. No forced batch/precision
changes, unrelated process signals or automatic failed-job retry. On failure,
drain peers, record error, stop admissions. Persistent tmux survives disconnect.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_mosei_nested_combinations_20261024.run \
  --source-commit SOURCE_SHA --output /absolute/new/execution_v1
```

Run from an immutable committed snapshot with SOURCE_SNAPSHOT.json. A repeated
directory needs explicit --resume and compatible complete checkpoint state;
completed/failed jobs never silently repeat. Actual startup is verified using
worker summaries and real epoch events, not inferred from tmux creation.
Four complete results generate RESULT.md/JSON and PER_RATE.csv, including the
three original controls and all mean/high ACC/WF1/MSE. No partial winner or
automatic three-seed promotion. Old seed66 Test-oracle gains are exploratory.
