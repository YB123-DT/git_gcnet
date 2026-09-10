# Complete-view Local-State JEPA: approved design and execution plan

User approved complete Local target instead of per-modality MMoE targets.
Student unchanged: observed mean encoder→causal OSRAM eta=.6 Flat→task head.
Teacher: deep-copied EMA observed_set and osram.local_path; full valid ATV input;
target = e_full + local_path(e_full). Eval/no-grad, no memory or label input.
Predictor: LN(hidden700), concatenate pattern embedding8x16, Linear716→256,
GELU, Linear256→256. No dropout, six-direction routing, InfoNCE or feedback.
Missing valid utterances only: mean SmoothL1(nonaffine LN(pred), detached
nonaffine LN(target)). Empty set returns differentiable zero. Weight .1 inherited.
EMA initialized exactly from Student, updated after optimizer step; buffers copied.
Default complete_state_jepa=False leaves old construction/RNG/state keys unchanged.
New mode initializes extra head under fork_rng, freezes legacy predictor/teacher.
All pre-existing shared parameters keep identical initialization to reference.

## Execution checklist

- Tests first: default compatibility; shape/missing mask; empty loss; no Teacher
  memory or future dependence; exact EMA update; strict state restore; gradient
  reaches predictor/Student/OSRAM not Teacher; no target leakage into Student.
- Implement gcnet_missing_m3/complete_state.py and model.py flag/helper integration.
- Add complete-state objective in train_gcnet.py; no old JEPA path, same student
  forward, state count/loss logged explicitly; train/evaluation compatibility.
- Run new and existing related tests in remote s0, one real MOSI batch backward.
- Launch five MOSI seeds66–70,100epochs,cyclic,eta=.6,Flat,per-rate Test-oracle.
  Inherit Joint Flat and Emotion-only scores from completed histories. No re-runs.
- Archive results, masks, config/provenance and per-rate epochs; upload GitHub.

INTERNAL DIAGNOSTIC ONLY, not formal paper results. No extra hyperparameter sweep.
Small state loss does not prove predictive sufficiency or prevent collapse.

## Verification and launch

4 component tests passed (shared init/RNG/forward, Teacher isolation, gradients,
EMA, strict state loading, seven patterns). 4 trainer tests and 28 existing
training lifecycle/CLI tests passed. Only existing PyG deprecation warning.
One real MOSI batch: state_loss=.724161, emotion_loss=2.844248, total=2.916664,
737 missing utterances. Nonzero finite predictor/encoder/OSRAM gradients,
no Teacher gradient; eval forward does not invoke Teacher. See SMOKE.json.
New trainable auxiliary parameters250872; new total parameters1312248 including
frozen EMA target modules. Full model trainable4853393; total8493465 includes
legacy frozen predictor keys retained for compatibility, not used for supervision.

Five runs launched on GPU2 seeds66/67/68 and GPU3 seeds69/70.
Remote root /data2/yb/remote_experiments/osram_complete_state_20260910.
Only model/config/selected-epoch weights are saved by inherited checkpoint code;
Teacher parameters are included. This is not an optimizer/RNG-exact resume facility.
EMA update counts are recorded in histories and final metrics, not a standalone
model-state buffer. Reconstructing the model requires mapping config objective
complete-state to constructor complete_state_jepa=True, as run_experiment does.
