# IEMOCAP-6 frozen Flat + cyclic Signature

INTERNAL TEST-ORACLE SCREEN. Finite user-authorized continuation of the completed
MOSI/MOSEI/MELD frozen-parent cyclic Signature experiments. This queue covers
IEMOCAP **six classes only**, seeds66/67/68, allfive LOSO folds, and all8 rates.
There are120 separate rate-specific stage2 branches,not15 single deployed models.
No original Flat is retrained and no four-class run or HPO is added.

## Fixed method and budget

- Each branch strictly restores its own accepted original Flat BEST for its
  dataset/seed/fold/evaluationrate. Accepted baseline registry:
  `experiments/osram_iemocap_flat_3seed_cmerc_20261016/SUMMARY.json`.
- Original parent training was100epochs,LR1e-3,batch32,Adam,WD1e-5,
  cmerc-repro and `gcnet-random-mask-v1`; it is not redefined here.
- Freeze original Encoder,Local,CHM,Query,Flat1600,LayerNorm and taskhead in eval.
  Train ONLY unchanged `SignatureBlock` (281856parameters),100epochs,
  constant branchLR3e-4,Adam,WD1e-5,batch32,original six-class CE.
- Use original `BalancedBatchRateSchedule`: cyclic0.0–0.7 during stage2,while
  evaluating and selecting only the designated current rate on its saved mask.
- Preserve host/guest sampling,padded rectangles,allnonempty-modality rules,
  causal read-before-write,joint Block Delta Write and write_step0.6.
- Signature uses strictly past8validutterances,visible modality latents,
  degree2 paths and zero output to the original pre-LayerNorm1600anchor.
  No-history/padding corrections are zero. No new losses,Gates or future input.

The accepted official LOSO protocol is reused exactly. It aliases validation to
the held-out session; this queue does not invent a new split or alter it behind
the user's back. TEST ACC selects each branch's earliest strictly best epoch;
UA,W-F1,CE and class recalls/support come from the same prediction and exact mask.
The result is internal exploratory Test-oracle evidence,not an independent
generalization claim. Additional stage2 training budget must remain disclosed.

## Parent and replay gates

Check registry identity,completed original Flat config,seed/fold,labelSHA,
feature roots,environment,checkpointSHA,selected epoch/rate and selected NPZ/mask
SHA. Old/current source differences only admit exact reviewed inactive extension
hash pairs; strict loading and bit-exact original logits/classes replay remain
mandatory. Unknown changes fail rather than replacing the parent or feature set.

Before trainer/config imports,bootstrap the original IEMOCAP datasetroot and
clear any inherited MELD override. Each normal run replays its accepted saved
mask with exactly one original scan perbatch,then checks zero-initialized branch
prediction equality. End-of-run gates require finite/nonzero Signature gradients,
unchanged frozen parent/no-grad,selected checkpoint replay,immutable source and
complete cyclic rate coverage. Original checkpoints/predictions are read-only.

## Persistent finite execution

From an immutable committed source snapshot on biggpu:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_signature_iemocap_cyclic_20261025.scheduler \
 --root /data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/execution_v1 \
 --source-commit ACTUAL_SOURCE_SHA \
 --gpu-map 0=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45,1=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049,2=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4,3=GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a \
 --per-gpu 6
```

One real seed66/fold5/rate.7 two-update GPU smoke gates formal admission.
Use UUIDbinding,measured VRAM/RAM plus reserve,one staggered admission perpoll,
up to6workers perhealthy card and24total. PhysicalGPU4 is forbidden; unrelated
GPU5/6/7 jobs are not modified. Busy resources cause waiting,not a changed batch,
model or missing protocol. All120 authorized tasks automatically fill vacancies.

The tmux-held controller writes QUEUE.json with PID/command/UUID/log/status.
Per-run config/history/summary,selected BEST/predictions,and complete last.pt
(branch,Adam,epoch,RNG) are preserved. Explicit `--resume` requires identical
source/identity and adopts live children without duplicate launch.

## Results

Final RESULT.md/JSON/CSV independently validate stored predictions and selected
metrics. Report each seed/fold/rate,fold eight-rate and high-rate means,then
equal-weight fivefold means following the baseline. Separately label pooled
out-of-fold prediction metrics; UA on concatenated predictions is not silently
substituted for average fold UA. Only complete fivefold seeds receive a seed
mean; three-seed sample SD is reported when allcomplete. All positive and
negative conditions remain visible. No incomplete or invalid run is a winner.
