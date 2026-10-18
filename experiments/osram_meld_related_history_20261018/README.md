# MELD B1/B2 related Local-history residual

INTERNAL TEST-ORACLE SCREEN. Two independent seed66 runs; no automatic HPO.

## Approved configuration

Original O6 Flat1600, Adam lr1e-5 and weight_decay2e-4,50epochs,batch32,
dropout0.5/projector_dropout0.1,cyclic0.0–0.7,gcnet-random-mask-v1.
Exact archived numerical implementation b4a313841f098487a6482d2712bdac94aa8e319e
is deliberately reused to compare to accepted seed66 O6 ACC61.786/UA38.435/WF158.918.
New branch source is versioned separately. This is from-scratch joint training
with the same original initialization, NOT loading O6 trained weights.
O6 and all protected baseline checkpoints remain unmodified.

## Computation

Only post-readout `flat_anchor → original emotion_norm` receives a residual.
The original adapter, unmodified Local Skip, emotion_norm and classifier stay.
No CHM/Query/K/V/writing/missing-rule changes; no second memory trajectory.

History candidates: previous actual post-local_path Local vectors, strictly
earlier and within8utterances, same conversation. Never unmasked raw features.
Attention128: q([L256,B512,availability3]), k/v(historyLocal256), dot/sqrt128.
Learned scalar score biases use same/different speaker, exact lag1..8 and
historical availability pattern encoded A/T/V bits1/2/4. Bias tables start0.
These metadata encodings are explicit first-version engineering choices.

B1 fusion input is `[L256,B512,GapA512,GapT512,GapV512,E128,availability3]`
**2435→128→GELU→1600**, last projection zero-init. No Gap compression/pooling.
Added parameters **682,834**, total **13,973,629**.

B2 retains the same retrieval and adds predicted consistency S:
`[Lt256,Lj256,sameSpeaker1,distance/window1,at3,aj3]`
**520→128→GELU→1→sigmoid**. E_same=sum R*S*v;
E_change=sum R*(1-S)*v; D=E_same-E_change; rho=sum R*S.
Fusion adds D128/rho1, **2564→128→GELU→1600**, finalzero-init.
Added parameters **766,163**, total **14,056,958**.

For B2 only, loss is original taskCE +0.05*relationBCE. Relation pairs are
all valid current/candidate pairs; same MELD class is1, otherwise0. BCE averages
within conversation then across conversations with valid pairs. No class
rebalancing. Main task uses differentiable S. Auxiliary recomputes S with
detached original representations: its direct gradients enter S predictor ONLY,
not Encoder/CHM/query/Flat/head or attention. Labels enter loss, never forward.
Regression helper supports sign pairs only when both abs(y)>0.5, not run here.

All padding/inactive Gap inputs use safe torch.where. Empty history residual
is hardzero even after outputbias learns. Original forward512 only is used;
the original full1024 context interface and backwardzero half remain untouched.
Branch initialization is RNG-isolated and branch forward has no added Dropout,
so original train/test mask streams can be verified against O6 each epoch.

## Execution

Use ssh biggpu, `/data2/yb/conda_envs/cmerc-repro/bin/python`.
GPU0 UUID `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`,twojobsparallel.
GPU4 forbidden; unrelated tasks untouched. Environment/data/source/hash checks
and same-forward best checkpoint+prediction/mask capture reuse sealed helpers.

```
python experiments/osram_meld_related_history_20261018/run.py \
  --worker --kind B1 --config /path/B1.json --output /path/smoke/B1 \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 --smoke
# Repeat B2; smoke has2trainbatches and1evalbatch/rate, NOT performance results.
python experiments/osram_meld_related_history_20261018/run.py \
  --output /path/runs_v1 --smoke-root /path/smoke
```

Resume controller only with --resume and same immutable snapshot/configuration.
Full last_training state contains model/optimizer/RNG/progress. Eight per-rate
BEST checkpoints preserve branch config plus original config and exact linked
selected-epoch predictions/masks. UA is from ACC-selected same epoch, not UA-best.
Controller writes RESULT.md/RESULTS.json after both finish, without extra search.

Run tests: `python -m pytest tests/test_related_history.py
tests/test_related_history_attachment.py tests/test_meld_related_history_runner.py -q`.
Smoke validates machinery, not improvement. Norms/consistency are not reliability,
causal support, true emotion shift or evidence of performance gains.
