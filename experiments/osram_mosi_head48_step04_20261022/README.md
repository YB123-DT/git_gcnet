# One MOSI66 head48 + write_step0.4 run

INTERNAL TEST-ORACLE SCREEN

Only one new from-scratch seed66 job. Key=Value=48,8heads,write_step0.4;
all original cfg84 no-JEPA Flat/Local Skip/CHM/query settings remain.
150epochs,LR3e-4 through100 then3e-5,Adam,batch32,WD1e-5,original dropout,MSE,
official MOSI split,s0 environment,GCNet random-mask-v1,cyclic rates0.0–0.7,
per-rate TEST W-F1-best (strict improvement,earliest tie).

Reuse the existing memory-grid training loop via explicit one-mode/seed hooks.
Fresh shared-shape parameters match original untrained seed66; step0.4 does
not change head48 initialization. Same original training mask/update/LR ledger
and exact common test masks checked. Save all eight BEST with matching
predictions/masks and complete last.pt (model/Adam/RNG/history).

biggpu only,healthy physical GPU2; require6GiB GPU/64GiB RAM/12GiB disk.
Forbidden GPU4 is checked by UUID. One persistent tmux worker; do not stop,
restart or alter the separate head48 seeds67/68 running on GPU7.
Reuse existing real head48 and write_step0.4 GPU validation; no extra smoke,
method, seed, loss or protocol change.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_mosi_head48_step04_20261022.run \
  --gpu-uuid GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
  --source-commit SOURCE_SHA --output /path/to/fresh/execution
```

After completing150epochs automatically write RESULT.md,REPORT.json,PER_RATE.csv,
comparing existing head64/step0.6,head64/step0.4,head48/step0.6 and the new
head48/step0.4. No new evaluation masks/inference or control training required.
Raw numerical core,environment,split,data,mask and seed provenance must match.
Head48 also shrinks Base/Gap/Flat input dimensions; internal Test-oracle and
candidate-selection bias remain. This single combination does not establish
multi-seed improvement. Keep every existing/new checkpoint and experiment.
