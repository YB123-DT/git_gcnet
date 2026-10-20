# MOSI/MOSEI 12-Source Evidence Audit

INTERNAL DIAGNOSTIC ONLY. Frozen inference, no optimizer, training or new module.

Use completed original Flat seed66 GCNet-missing runs: MOSI150 (`execution_v2`
of osram_mosi_gcnet_testbest_20261020) and MOSEI60 (`execution_v1` of
osram_mosei_gcnet_router_20261020). Each dataset uses its single miss=.3 BEST
checkpoint for all conditions. Never use the newly running LR1e-5 models,
Router weights, old conversation-mask-v2 models, or different weights per rate.
Thus full-model scores here are not the eight-rate Test-best baseline mean.

Replay exact saved original GCNet masks for random0.0–0.7, then separately
evaluate Persistent A/T/V/AT/AV/TV/ATV. No resampling, mask algorithm changes,
test threshold search, feature extraction or label-dependent routing.
Strict-load checkpoint, verify source/data hashes, test cohort and exact
saved miss=.3 prediction before capture. Checkpoint and existing results remain
read-only. Preserve original s0 Torch2.2.2+cu121/NumPy1.24.3/sklearn1.3.2.

Rows: Base/Gap-A/Gap-T/Gap-V. Columns: historical A/T/V joint-write Value-slot
sources. For each condition scan original CHM once per batch, capture the actual
Flat Local and memory input, then replay only original readout for12 independent
single-source removals and8 source coalitions. Original K/V,queries,joint write,
memory trajectories,LocalSkip,head and weights do not change. Source tracker
uses existing algebraic joint-write decomposition; not pure modality semantics.

Each 4x3 cell reports ACC and W-F1 after removing that source, plus difference
from original on the SAME active-row cohort. Gap rows only include currently
missing target modalities; inactive rows are N/A. Report active N and polarity N.
For MOSI/MOSEI retain original gold!=0 filtering and prediction>0 threshold;
MSE uses all valid utterances. No categorical argmax, UA or CE is used.
Correction/harm means removal fixes/introduces polarity errors. Positive
ACC/W-F1 delta means removal improves the fixed classifier, not that its
underlying historical modality is inherently harmful. Independent12cell effects
are not additive; grouped Shapley uses payoff=-MSE, never per-sample W-F1.

Run two finite jobs sharing healthy GPU3 (actual UUID verified at launch),
not GPU2's existing LR experiment. GPU4 forbidden. Require32GiB RAM,12GiB disk
and12GB free VRAM. Each job first runs one real batch to check scalar readout,
source sum, zero/inactive slots, frozen state and RNG, then all15 conditions.
No backward/optimizer or new checkpoint writes. Large NPZs stay remote.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u -m \
  experiments.osram_scalar_12_source_audit_20261020.launch \
  --source-commit SOURCE_SHA --gpu-uuid HEALTHY_GPU_UUID \
  --output /data2/yb/remote_experiments/osram_scalar_12_source_audit_20261020/execution_v1
```

Per dataset outputs: ACC_PER_RATE.md,WF1_PER_RATE.md,PERSISTENT.md,RESULT.md,
CONDITIONS.csv,SLOTS.csv (corrections/harms included),SLOTS_GROUPED.csv,
SHAPLEY.csv,REPORT.json. Reports only accept all15complete conditions and
unchanged frozen backbone. No automatic extension to other seeds or training.
