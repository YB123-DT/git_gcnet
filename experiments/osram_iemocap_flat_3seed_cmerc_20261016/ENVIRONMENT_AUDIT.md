# Accepted comparison environment audit

Verified read-only on biggpu,2026-10-16, before the additional Flat seeds.
This audit does not authorize or launch new MELD work.

## Runtime used by the current comparisons

Executable: `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Python3.10.13; PyTorch1.13.1+cu117; Torch CUDA11.7; cuDNN8500;
NumPy1.26.4; scikit-learn1.3.2. The new replication launcher and worker must
reject another interpreter or Torch/CUDA/NumPy/sklearn version before training.

| Evidence group | What was verified |
|---|---|
| New IEMOCAP Flat66 | Completed10/10folds. Manifest and saved queue commands name cmerc-repro; per-fold provenance reports Torch1.13.1+cu117 and NumPy1.26.4. |
| IEMOCAP B6 | Completed30/30folds. Manifest executable is cmerc-repro; all30 summaries record the same Torch/NumPy versions; queue commands name that executable. |
| MELD matched Flat/B6 replication | All seven new queue commands name cmerc-repro and original worker provenance records the same Torch/NumPy versions. Reused B6seed66 provenance also matches. |
| Accepted MELD original Flat60.693 | BASELINE.md records cmerc-repro and these versions, protected24BEST weights and exact selected-mask evidence. Baseline directories exist; their per-run PROVENANCE does not itself contain full environment fields. Do not imply a new audit recovered missing historical package snapshots. |

MELD's original worker summaries preserve post-training common-mask replay
failures. Training and requested Test-best metrics were separately recovered
and verified under `testbest_recovery_v1/SUMMARY.json`, whose status is
complete. This does not repair or complete the common-mask replay and does
not mean that six training jobs need to be repeated.

Historical IEMOCAP Flat at `b4a3138` used the `adapter` interpreter, not this
environment. Its results remain archived separately; they are not the Flat
reference for the new cmerc-repro three-seed comparison.

## Finite authorized completion

- Reuse existing Flat66: two class settings × five folds =10complete jobs.
- Train only Flat67/68: two settings × two seeds × five folds =20new jobs.
- Reuse existing B6: two settings × three seeds × five folds =30complete jobs.
-100epochs, original GCNet random missing, cyclic0.0–0.7, original Flat and
  Local Skip, unchanged IEMOCAP training hyperparameters and numerical source
  `785a01242ef1a82b97d5337acbb33ba963df5537`.
- Per-rate Test-best ACC; calculate UA from the same selected predictions.
  INTERNAL TEST-ORACLE DIAGNOSTIC, not independent test evidence.
- Do not rerun Flat66, B6, MELD or another seed; do not delete old results.
- Keep every BEST, matching selection artifact, and complete last-training
  state. Reusing the completed66 files means references, not overwriting them.

The different datasets retain their own documented task configurations.
Sharing an environment does not justify copying MELD learning rate, dropout
or weight decay to IEMOCAP, or claiming exact training determinism.

Source records:

```
/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016/{MANIFEST.json,queue/QUEUE.json,runs/}
/home/qinjinghui/remote_experiments/osram_iemocap_b6_transfer_20261015/{MANIFEST.json,queue/QUEUE.json,runs/}
/home/qinjinghui/remote_experiments/osram_meld_b6_top2_matched_flat_3seed_20261015/{queue/QUEUE.json,runs/,testbest_recovery_v1/SUMMARY.json}
experiments/osram_meld_testbest_ua_20261014/BASELINE.md
```
