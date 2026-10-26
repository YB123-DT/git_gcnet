# Seeds67/68 frozen Nested: read-only preflight review

Audit date: 2026-10-26 UTC. Server: `ssh biggpu`. This review did not start,
stop or modify training. The only local review artifact is this document.

## Verified scope and artifacts

Exactly128 new branches: each seed67/68 has MOSI8, MOSEI8, MELD8 and
IEMOCAPSix five LOSO folds ×8rates. The existing seed66 Nested queues are
complete: MOSI8/8 and cross-dataset56/56. Their64 branches are reused.

The live read-only audit recomputed all128 protected parent checkpoint SHA256s,
all128 selected-prediction SHA256s and all128 selected availability Mask SHA256s.
Every digest matched the protected registry or completed same-seed Signature
identity. No failure was observed. Exact command output:

```json
{"checkpoints_checked":128,"predictions_and_masks_checked":128,"bytes_hashed":10615322303,"seconds":30.8896965989843,"errors":[]}
```

The parent summaries/configurations are complete and retain their original
seed/fold, `gcnet-random-mask-v1`, cyclic training, batch32, Adam and WD1e-5.
Their recorded runtime is cmerc-repro: Torch1.13.1+cu117, NumPy1.26.4 and
sklearn1.3.2. This is artifact/config verification; new wrapper prediction replay
and GPU smoke remain execution-time gates.

| Dataset | Parent location / registry | Parent budget and LR |
|---|---|---|
| MOSI67/68 | `/data2/yb/remote_experiments/osram_cmerc_sentiment_3seed_20261023/execution_v1/runs/CMUMOSI/seed_SEED/flat` | 150epochs,3e-4 then3e-4/10 |
| MOSEI67/68 | Same root,`CMUMOSEI/seed_SEED/flat` | 60epochs,constant1e-4 |
| MELD67/68 | `/home/qinjinghui/remote_experiments/osram_meld_fixed_lr_wd_20261018/runs_v1/SSEED_WD1`; protected `osram_meld_wd_seed66_20261018/BASELINE.json` | 50epochs,constant1e-5 |
| IEMOCAPSix67/68 | `/home/qinjinghui/remote_experiments/osram_iemocap_flat_3seed_cmerc_20261016/runs/iemocap6/seed_SEED/fold_FOLD`; accepted three-seed registry | 100epochs,constant1e-3 |

MELD67/68 summary SHA256s also match the protected baseline registry. All80
IEMOCAP selected-prediction hashes were independently checked before the full
128-artifact audit and passed.

## Source ownership and trace boundaries

All numerical roots below already exist under
`/data2/yb/remote_experiments/` and must remain immutable.

| Conditions | Numerical snapshot | Completed Signature reference |
|---|---|---|
| MOSI67; MOSEI67/68 | `osram_signature_cyclic_multi_20261025/source_e0ca94c` | `osram_signature_cyclic_multi_20261025/execution_v2/train/DATASET_SEED_RATE` |
| MOSI68 | `osram_signature_cyclic68_20261025/source_4d5aa09` | `osram_signature_cyclic68_20261025/execution_v1/train/RATE` |
| IEMOCAPSix67/68 | `osram_signature_iemocap_cyclic_20261025/source_21e46f2` | `osram_signature_iemocap_cyclic_20261025/execution_v1/train/IEMOCAPSix_SEED_foldFOLD_RATE` |
| MELD67/68 | Existing `source_e0ca94c` is the candidate preparation bridge; new per-seed preparation/replay must validate it | No completed same-seed Signature reference found |

All112 applicable same-seed Signature references are complete and have the
correct seed/source/budget plus per-epoch mask digests and optimizer-step
counts. MOSI68's rate-only directory layout is intentional.

MELD67/68 retain50epoch `MASK_TRACE.json`, `history.json` and
`OPTIMIZER_HISTORY.json` in their protected Flat runs. These are Flat training
records, not completed frozen-stage2 Signature traces. Their hash format and
RNG context cannot silently be substituted for stage2 `(conversation_ids,
availability)` traces. Do not reuse seed66 traces as67/68 evidence. An independent
same-seed sampling audit may establish the new stage2 sampling contract without
training an additional comparison model; label the evidence accurately.

MOSEI67/68 keep their own parent weights and the accepted fixed test-mask bank
referenced from seed66, following existing `prepare.mask_path` semantics.

The following five Nested implementation files have identical SHA256s across
all three numerical snapshots (`e0ca94c`, `4d5aa09`, `21e46f2`):

| File under `gcnet_missing_m3/` | SHA256 |
|---|---|
| `meaningful_input.py` | `e4a86fe4aabd42ca9d0b2ccd503e9d588afa9f6f803d8e74748fe6970c1037f8` |
| `meaningful_input_new40.py` | `cf97479e9190e78f26fcf373c8c4a4b8ea5d4ebd4b85ec7ece7ef45345fb84fe` |
| `meaningful_blocks_common.py` | `8647e22cbff8fdf851c72951de0253ed2c419c334e732f029ea6d30406491766` |
| `meaningful_new40_structure.py` | `4980e2e8eb75442a1096118232973d5867531524ceb4d5568bf85674d5bffa8a` |
| `meaningful_new40_registry.py` | `5491725bd76a1de571a668f921e214de1966b8904851f868d5790f01d1055467` |

## Resource snapshot and reusable measured cost

Read-only resource commands:

```bash
ssh biggpu 'nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits; free -h; df -h /data2/yb/remote_experiments'
```

| Physical GPU | UUID | Free MiB | GPU utilization | Compute processes |
|---|---|---:|---:|---:|
| 0 | `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45` |32495|0%|0|
| 1 | `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049` |32495|0%|0|
| 2 | `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4` |32495|0%|0|

All three are V100-SXM2-32GB. RAM available460GiB; `/data2` free283GiB,
92%used. GPU4 remains forbidden irrespective of its query result. GPU5/6 were
busy with unrelated jobs and GPU7 had an unrelated process; none were touched.
These values are a point-in-time admission input, not a reservation.

Completed seed66 formal-run measurements, obtained directly from per-run
`SUMMARY.json` files:

| Dataset | Runs | Maximum allocated GPU MiB | Maximum RSS GiB | Per-run seconds range |
|---|---:|---:|---:|---:|
| MOSI |8|1284.9453|2.3647|415.40–475.88|
| MOSEI |8|980.3335|3.8519|4577.20–5429.42|
| IEMOCAPSix |40|2576.1284|2.6715|719.55–1001.62|
| MELD |8|595.5347|2.6898|1732.78–1954.10|

The existing159235-parameter wrapper and these full-run resource measurements
can be reused. Keep CUDA-context/allocator and live-system reserves, stagger
admission, and observe actual throughput; concurrent historical wall times are
not isolated speed benchmarks.

## Reproducible artifact-audit command

This is the command used for the successful full128 audit. It reads protected
files, hashes them and prints one small result; it creates no remote artifact.

```bash
ssh biggpu 'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /data2/yb/conda_envs/cmerc-repro/bin/python -' <<'PY'
import hashlib,json,time
from pathlib import Path
import numpy as np
R=Path('/data2/yb/remote_experiments');count=0;pred_count=0;total=0;errors=[];start=time.monotonic()
def sha(path):
 global total
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):total+=len(b);h.update(b)
 return h.hexdigest()
for d in ('CMUMOSI','CMUMOSEI','IEMOCAPSix','MELD'):
 for s in (67,68):
  for f in (range(1,6) if d=='IEMOCAPSix' else (1,)):
   for k in range(8):
    rate=str(k/10)
    try:
     if d=='MELD':
      registry=json.loads((R/'osram_signature_cyclic_multi_20261025/source_e0ca94c/experiments/osram_meld_wd_seed66_20261018/BASELINE.json').read_text());run=next(j for j in registry['runs'] if j['seed']==s);p=Path(run['output']);b=run['testbest']['rates'][rate];cp=p/f'best_miss_{rate.replace(".","p")}.pt';ch=b['checkpoint_sha256'];pp=p/b['path'];ph=b['sha256'];mh=b['availability_sha256']
     else:
      if d=='CMUMOSI' and s==68:p=R/'osram_signature_cyclic68_20261025/execution_v1/train'/rate
      elif d=='IEMOCAPSix':p=R/'osram_signature_iemocap_cyclic_20261025/execution_v1/train'/f'{d}_{s}_fold{f}_{rate}'
      else:p=R/'osram_signature_cyclic_multi_20261025/execution_v2/train'/f'{d}_{s}_{rate}'
      ref=json.loads((p/'SUMMARY.json').read_text());parent=ref['identity']['parent'];cp=Path(parent['checkpoint']);ch=parent['checkpoint_sha256'];mh=ref['identity']['test_mask_sha256']
      if 'selected_prediction' in parent:pp=Path(parent['selected_prediction']);ph=parent['selected_prediction_sha256']
      else:
       b=json.loads((Path(parent['parent_path'])/'SUMMARY.json').read_text())['best_by_rate'][rate];pp=Path(parent['parent_path'])/b['predictions'];ph=b['prediction_sha256']
     assert sha(cp)==ch;count+=1
     assert sha(pp)==ph
     with np.load(pp,allow_pickle=False) as a:assert hashlib.sha256(a['availability'].tobytes()).hexdigest()==mh
     pred_count+=1
    except Exception as e:errors.append((d,s,f,rate,repr(e)))
print(json.dumps(dict(checkpoints_checked=count,predictions_and_masks_checked=pred_count,bytes_hashed=total,seconds=time.monotonic()-start,errors=errors)))
PY
```
