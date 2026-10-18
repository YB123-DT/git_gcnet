# B1/B2 frozen contribution, epoch selection and S diagnostic

INTERNAL DIAGNOSTIC ONLY. No new training, seeds, U model or mask sampler.

Source B1/B2 seed66 are completed50epoch O6 runs (lr1e-5,wd2e-4), archived
training extension442596a, original numerical b4a3138, cmerc-repro environment,
accepted ECERC MELD feature hash and gcnet-random-mask-v1. Source BEST states,
selected-epoch prediction NPZ, full recovery states and training versions are
opened read-only; all read input hashes rechecked after the diagnostic.

## Three questions only

1. Each8rate BEST in B1/B2: replay exact original selected mask and metadata.
   One original CHM scan per batch, capture actual preLN Flat anchor and new
   residual. Readout on/off shares all parameters, original norm/head and
   unchanged L/B/G. Save CE/ACC/UA/W-F1, corrections/harms, residual norm and
   residual/anchor norm ratio. Off is not separately trained original Flat.
2. At rate0.1 use B1 epoch13 exact selected-mask as common input. Reuse B1
   output from question1. Replay B2 selected epoch10 and epoch13 from rate0.5
   BEST on that SAME mask. Verify source summary, config, final-state training
   identity, selected metadata and semantic state-dict equality against the
   committed training version. Thus epoch13 is not from another model/run.
   Report B1e13→B2e13 (sameepoch), B2e10→B2e13 (selection), B1e13→B2e10.
3. B2 original forward exposes actual pair logits/S/R. Only valid pairs from
   same conversation,1≤lag≤8 and real utterances enter offline same-class
   scoring. No gold label enters model or branch forward. Fixedthreshold0.5,
   ten equalwidth calibration bins, AUROC/AUPRC,balancedACC,positiveprecision/
   recall/F1,BCE,Brier,ECE,probability distribution and prevalence are reported.
   Main scores unweighted, retrieval-R-weighted scores secondary. No fitting.
   Frozen S=.5 replay edits only final fusion D128→0 and rho→.5 when history
   exists; E,R,L,B,G unchanged, hardzero residual for empty history retained.
   No second Memory scan or new Query is needed for any condition.

18 finite dataset passes:16 original selected B1/B2 rate-checkpoints plus2
commonmask B2 checkpoints; B1 commonmask pass reused. All inference frozen.
No checkpoint search or alternate thresholds. Individual prediction/pair NPZ
stay in new remote output, not Git. Report/JSON are small tracked artifacts.

Runtime: ssh biggpu, `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Healthy physicalGPU3 UUID GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a,logicalcuda:0.
GPU4 forbidden, other training tasks remain untouched. One inference process.
Immutable diagnostic snapshot copies original sealed helpers/extension, then
adds this directory and the reused small-LR readout metadata helper. Both
original and diagnostic manifests checked before executing. No original files
or numeric implementation are edited.

```
python experiments/osram_meld_related_history_diagnostic_20261018/run.py \
 --output /home/qinjinghui/remote_experiments/osram_meld_related_history_diagnostic_20261018/runs_v1 \
 --gpu-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a \
 --diagnostic-commit <actual-committed-diagnostic-source>
```

Tests: `python -m pytest tests/test_related_history_diagnostic.py
tests/test_related_history_statistics.py tests/test_related_history_diagnostic_runner.py -q`.
36 tests cover actual attached OSRAM, single scan, exact selected-prediction
parity, half-intervention semantics, mask/padding/exception cleanup, relation
metrics, trajectory identity and absence of weight/RNG changes.

## Limits

Whole-residual contribution cannot by itself isolate whether E or the Local/
Base/Gap already inside the residual MLP caused any gain. S=.5 is sensitivity
to an out-of-training-distribution internal condition, not a trained model's
benchmark. Sameclass predictability is not task usefulness or causal support.
Repeated pairs across8rates are not independent samples. No statistical
significance, reliability, real emotion shift or new model performance claims.
