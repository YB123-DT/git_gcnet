# Direct source weighting: finite cross-dataset confirmation

INTERNAL TEST-ORACLE SCREEN

User-authorized scope: MOSEI and MELD seeds66/67/68, plus IEMOCAP-6 all five
LOSO folds for each seed. Exactly21 fresh one-stage training runs; no new MOSI
runs, Flat reruns, HPO, second-stage freezing or additional methods.

This is the existing `CoupledSourceRole(mode='direct')`,96,487 added parameters:
`gamma=2*sigmoid(shared_partner_read_MLP)`, zero-initialized last score layer.
Base/each active Gap becomes the sum of its three gamma-weighted joint-Value
source reads, WITHOUT adding back the original Base/Gap. The ORIGINAL Local
Skip remains. CHM queries,K/V,joint Block Write,read-before-write,write_step.6,
8heads×64,Flat1600,norm/head and task objectives remain unchanged. No completion,
JEPA,probability Router,persistent view or auxiliary loss. Sources are joint
Value-slot contributions, not three independent pure-modality memories.

| Dataset | Seeds/folds | Epochs | Adam LR | WD/batch | Runtime | Per-rate selection |
|---|---|---:|---:|---|---|---|
| MOSEI |66/67/68;fold1|60|1e-3|1e-5/32|archived s0,Torch2.2.2+cu121|TEST W-F1 |
| MELD |66/67/68;fold1|50|1e-5|1e-5/32|cmerc-repro,Torch1.13.1+cu117|TEST ACC |
| IEMOCAP-6 |66/67/68;folds1..5|100|1e-3|1e-5/32|cmerc-repro,Torch1.13.1+cu117|TEST ACC |

Constant learning rates. Class UA/W-F1/CE/recall/support come from the ACC-selected
same prediction/mask. Earliest strict-best ties; no validation selection. Eight
BEST checkpoints and a complete model/Adam/RNG/history `last.pt` are retained.
Per-rate means may use different epochs: not a single deployment checkpoint.

All use original `gcnet-random-mask-v1`, host/guest,padded rectangles,cyclic0–.7.
MELD retains its original per-epoch resampled test masks using the original
trainer's no-grad `evaluate_rate`; it does NOT silently switch to a fixed bank.
MOSEI/IEMOCAP retain their existing isolated seed66 fixed legal GCNet banks;
IEMOCAP uses all four training sessions, no additional adapter holdout. Test
identities/speakers/availability are saved with every selected prediction.

Original Flat references are read-only: MELD baseline registry
`../osram_meld_wd_seed66_20261018/BASELINE.json` (61.557ACC/37.728UA);
IEMOCAP `../osram_iemocap6_gcnet_testbest_20261020` (100epochs,not150);
MOSEI `../osram_mosei_gcnet_router_20261020` (only seed66 currently paired).
No nonexistent MOSEI67/68 Flat reference is invented. The final report does
not silently claim same-mask/trajectory pairing with old independently selected
checkpoints. No protected weights initialize training or are overwritten.

## Execution

On `ssh biggpu`, user specifically requested GPU1 concurrency. Bind the actual
UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`; physical GPU4 is forbidden.
Run from a committed immutable snapshot with independent output paths:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
 experiments.osram_source_role_direct_cross_20261020.run \
 --source-commit SOURCE_SHA \
 --output /data2/yb/remote_experiments/osram_source_role_direct_cross_20261020/execution_v1 \
 --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 --max-parallel 6
```

Three short dataset/runtime GPU smokes check initial prediction tolerance,
finite/nonzero source-role gradients, backbone/new-module updates and full
checkpoint reload. Formal workers then run concurrently up to6, with1.5×
measured GPU peak reservations plus2GiB/card, availableRAM>=64GiB,48GiB disk
preflight. No peer kill or batch/architecture reduction to fit. Persistent
tmux controller fills freed slots automatically. Every dataset child uses its
own pinned interpreter. Failed children stop new admissions; running peers
are left alive. Explicit same-source `--resume` requires full compatible state
and refuses duplicate live PIDs or completed jobs.

Outputs: QUEUE.json,isolated per-runlogs,config/source/data/environment/GPU
identity,epochmetrics,matching per-rateBEST+predictions/masks,last.pt. Final
RESULT.md/METRICS.csv/REPORT.json only after all21full runs succeed. IEMOCAP
aggregates equal fold means within seed/rate,then eight-rate means withinseed,
then three-seed means and sampleSD; never mixes pooled metrics silently.

## Verification boundary

This launcher reuses the already-tested direct weighting and tracker exactly;
no numerical core files change. Initial output matches fresh Flat within the
previously declared1e-5 absolute/relative tolerance (source summation can round).
Tests/real smoke verify implementation, not predictive benefit. No claim of
independent-test generalization or guaranteed improvement is made.
