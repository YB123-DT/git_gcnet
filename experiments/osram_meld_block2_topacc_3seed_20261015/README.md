# MELD B6 top Test-best ACC: three-seed replication

INTERNAL TEST-ORACLE THREE-SEED REPLICATION.

User authorization: replicate the better completed Test-best ACC configurations
with three seeds. Freeze the three highest completed configurations at the
authorization snapshot; do not keep replacing them as the independent B2 search
progresses. All three currently belong to B6 Learned Robust PCA, not three
different architectures.

| Configuration | Seed66 Test-best ACC | learning rate | weight decay | dropout |
|---|---:|---:|---:|---:|
| B6 trial16 | 60.914751 | 0.0003433234536467373 | 0.0000044857894231052665 | 0.22489055023388654 |
| B6 trial13 | 60.909962 | 0.0006385962773421361 | 0.00010907264837159956 | 0.20816400308761954 |
| B6 trial12 | 60.890805 | 0.001381200239152821 | 0.000013377383604517501 | 0.20402927084751332 |

Reuse the three completed seed66 runs from immutable source `7abb855`. Add
seed67 and seed68 for each fixed configuration, six new from-scratch jobs total.
Keep 50 epochs, batch32, cyclic rates0.0–0.7, GCNet random-mask-v1,
Test-oracle-per-rate ACC checkpoint selection, exact selected-epoch predictions
and masks, original CHM/Flat/Local Skip, and all original losses unchanged.
Do not average UA-selected epochs or final randomly re-evaluated predictions.

`MANIFEST.json` pins effective configurations, original source summaries and
their hashes, source/data/mask hashes, seed66 metrics, and measured smoke costs.
The replication launcher verifies that model/trainer/sampler source hashes match
those original runs. The worker's only control-flow extension is explicit seed
admission; without `--replication-seed`, it retains its seed66-only guard.
No model code or numerical training code was changed for replication.

Server: `ssh biggpu`; Python `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Use healthy host GPUs0/2/3 via real UUIDs, never GPU4. New replication queue
uses at most two processes/card, six total, alongside at most two existing HPO
workers. Preserve the original HPO process and all historical checkpoints.
New outputs use the same server's root filesystem to avoid `/data2` exhaustion:
`/home/qinjinghui/remote_experiments/osram_meld_block2_topacc_3seed_20261015`.

Run from the new immutable source snapshot:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_meld_block2_auto_20261014.replicate \
  --manifest experiments/osram_meld_block2_topacc_3seed_20261015/MANIFEST.json \
  --output /home/qinjinghui/remote_experiments/osram_meld_block2_topacc_3seed_20261015 \
  --gpus 0,2,3 --per-gpu 2
```

Persistent nohup execution writes `controller.log`; queue state/PIDs/logs/configs
are under `queue/`. Each job retains eight BEST checkpoints and exact selected
artifacts, with epoch-boundary optimizer/RNG recovery. Automatic final outputs:
`RESULT.md`, `SUMMARY.json`, `SUMMARY.csv`, including per-rate ACC/UA/W-F1,
eight-rate/high-missing means and three-seed sample SD.

Seed66 selected these configurations; the mean including it is exploratory, not
an independent test estimate. The accepted historical Flat three-seed mean is
reference only, not a new same-snapshot three-seed control. This task does not
authorize additional Flat runs or another hyperparameter search.
