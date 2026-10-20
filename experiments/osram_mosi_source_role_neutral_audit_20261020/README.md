# MOSI three-seed12-slot neutral-weight audit

INTERNAL DIAGNOSTIC ONLY

Scope: existing direct-source150epoch seeds66/67/68,all eight rates. Use each
rate's already selected BEST checkpoint and exact saved GCNet evaluation mask.
No training,checkpoint selection,mask sampling,optimizer,weight update,threshold
search or new model module. Never modify protected weights or predictions.

Capture Local,original Flat input,12learned gamma and joint Value-slot source
reads from ONE original causal CHM scan per batch. Reuse original Local Skip,
Adapter,LayerNorm and scalar task head for14readouts:learned,all active gamma=1,
and12individual slots reset to1. No second Memory/query trajectory.
Gamma1 means neutral calibration,NOT gamma0 source deletion. All inactive
Gap/source/first-utterance/padding contributions stay inactive. Backward half
remains zero. Original unmodified Flat predictions are reused from stored
same-seed/rate artifacts,not re-inferred with fresh masks.

Primary cells:W-F1(learned)-W-F1(one slot gamma1),in percentage points;positive
means the learned adjustment helped this fixed classifier. Three4x3tables
average full-test changes equally across eight rates. Per-rate active-cohort
metrics,N,corrections/harms and gamma stats are separate; inactive subsets N/A.
Do not sum12nonlinear marginal effects or reconstruct full W-F1 from subgroup
averages. Counts across rates are repeated exposures,not unique corpus totals.

All-gamma1 uses the NEW trained checkpoint,not original Flat. Compare total
learned−Flat=(neutral−Flat)+(learned−neutral),on each same cohort/metric. This
is a behavior decomposition,not proof of a pure semantic cause of training.
Source labels remain joint-write Value-slot terms,not pure modality memory.

Strict checkpoint/data/numerical-source hashes,original s0 environment,cohort
and availability order are checked. Learned predictions must exactly reproduce
all24saved predictions; weighted source reconstruction and original readout are
checked exactly. Neutral reconstruction of original raw reads is only within
the existing floating-point source decomposition tolerance. RNG and frozen
model digest must not change. No optimization/gradient call is made.

Run on biggpu healthy physicalGPU0,never badGPU4. One serialized audit job;
RAM>=32GiB,disk>=4GiB,original batch32. First real batch invariance is embedded
in the actual diagnostic; no duplicate formal smoke/training job.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
 experiments.osram_mosi_source_role_neutral_audit_20261020.run \
 --output /data2/yb/remote_experiments/osram_mosi_source_role_neutral_audit_20261020/execution_v1 \
 --source-commit SOURCE_SHA --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
```

Source is immutable under persistent tmux. Outputs SUMMARY.json,24NPZ/JSON
conditions,RESULT.md,PER_RATE.md,SLOTS.csv,METRICS.csv,REPORT.json. Large NPZs
stay remote; only code,small machine-readable statistics and reports go toGit.
No automatic training or further experiments after the complete24conditions.

Prelaunch:29focused neutral-readout/analysis/direct/three-seed tests passed;
new readout and analysis tests failed first before implementation. Compileall
and git diff --check passed. Existing unused optional torch-sparse import
warning is not an operator used by this audit. No original model code changed.
