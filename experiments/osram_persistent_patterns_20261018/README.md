# Frozen Persistent7: MELD / IEMOCAP4 / IEMOCAP6

INTERNAL DIAGNOSTIC ONLY. No training, new model, changed numerical path or
checkpoint reselection. This is the user's continuation of persistent modality
evaluation, NOT another random0.0–0.7 training run.

Fixed observed sets: A,T,V,AT,AV,TV,ATV. T=Text. Every valid utterance and its
history in each test conversation receive exactly this support; host/guest
masks agree and padding stays zero. No missing features are exposed.

Pre-fixed checkpoint mapping: singleton=.7 BEST; pair=.3 BEST; ATV=.0 BEST.
These were chosen previously under original per-rate Test ACC, not by new
Persistent scores. Source weights/selection artifacts opened read-only and
hashed; exact original selected masks/logits/classes must replay first.
All model parameters frozen, original Memory/Query/Local Skip/Flat unchanged.

MELD: accepted lr1e-5/wd1e-5/50epochs,Flat1600,3seeds66/67/68. A/T/V already
completed in osram_meld_persistent_unimodal_20261018 are reused with artifact
verification, not recomputed. New inference only pairs/full and their sanity
replays. Original numerical b4a3138 and cmerc-repro remain unchanged.

IEMOCAP4/6: existing original Flat1600,cmerc-repro,100epochs,lr1e-3/wd1e-5,
new GCNet-random-mask-v1. Complete3backbone seeds ×5LOSO folds each; source
785a012. Not older environment, Nested, B6, fifth-fold-only or150epoch runs.
Constructor-expanded checkpoint config must exactly match retained config.json.
Average five folds equally within seed, then three seeds; no utterance pooling.

Report ACC,UA,W-F1,CE,recall/support; UA uses actual4/6/7classes. Also report
six nonfull and seven-pattern macro averages, and separately the previously
completed random eight-rate reference. Do not mix the two evaluation conditions.
Per-seed/fold/pattern CSV and per-seed fivefold-mean CSV are exported.

Three frozen inference workers share healthy physicalGPU3 UUID
GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a. Earlier real MELD inference peaked
at205MiB; admission retains ample headroom for the two IEMOCAP workers.
GPU4 forbidden; no other user's tasks signalled. Runtime:
/data2/yb/conda_envs/cmerc-repro/bin/python,Torch1.13.1+cu117,NumPy1.26.4,
sklearn1.3.2. Pin an immutable source manifest before launch; no new dependencies.

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u \
 experiments/osram_persistent_patterns_20261018/run_all.py \
 --output /home/qinjinghui/remote_experiments/osram_persistent_patterns_20261018/runs_v1 \
 --commit <actual-committed-source>
```

Use tmux. Independent per-dataset logs/results; all three workers must complete
before final aggregation. Failed/partial folds cannot appear in a mean. No
automatic retries, altered masks or extra experiments. Checkpoints, dense data
and prediction NPZ stay remote; commit only code, text and small result records.

Tests: test_persistent_pattern_evaluate.py, test_persistent_pattern_analysis.py,
test_persistent_iemocap_source.py; no extra training smoke. Original all-visible
selected replay is identical to ATV and reused, avoiding a duplicate forward.
