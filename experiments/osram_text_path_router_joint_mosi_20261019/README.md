# MOSI one-stage historical Text-source Router

INTERNAL EXPLORATION; seed66 only, with a matched fresh Flat control.

User reference “Flat80.7” means the historical Flat150 three-seed W-F1 mean
80.747%, not seed66. Historical seed66 is81.068095% (high missing76.352251%).
See `experiments/osram_complete150_20261011/RESULT.md`.

Retain the actual original MOSI data/features, environment and missing sampler:
`/data2/yb/reproduction_workspace/envs/s0/bin/python`, Torch2.2.2+cu121,
NumPy1.24.3, sklearn1.3.2, `conversation-mask-v2`, original frozen evaluation
masks. Do NOT substitute the MELD cmerc-repro environment or GCNet mask.
Train150 epochs:100 at original backbone1e-3, then50 at1e-4 without resetting
Adam/RNG/model or freezing anything. The Router LR is3e-4 then3e-5. Batch32,
Adam, wd1e-5, original dropout and scalar MSE are retained.

The historical trainer intentionally/historically derives model initialization
from `missing_m3_model_init:fold:5` even for MOSI. Preserve that actual RNG label;
data fold remains1. Current-core loading of the protected old .3 checkpoint must
strictly reproduce its original saved scalar predictions and masks before a
fresh current-core experiment is allowed. Never initialize training from that
protected checkpoint, silently fall back to another source, or overwrite it.

Architecture and trigger match the classification Router: current Text missing,
strictly-past Text observed; one original Memory scan; three shared Flat/head
readouts (full, omit Gap-T←T, omit Base←T), same Dropout and preserved original RNG.
For MOSI only, normalize Router weights and mix the three **raw scalar scores**.
Do not Softmax the one-dimensional predictions: that would be constant1 and
destroy the original MSE task. All original and new parameters train jointly.

Like the current MELD/IEMOCAP round, select ONE checkpoint using validation:
mean eight-rate W-F1, ties lower MSE. Replay original fixed test masks after
selection. Thus old per-rate Test-best81.068/80.747 are historical references,
NOT strictly matched scores; compare against the fresh same-protocol Flat.
Do not hide this selection difference or claim a paper benchmark improvement.

The finite orchestrator runs one real-data joint smoke, then exactly Flat+Joint
seed66 on one verified healthy GPU. No MOSI seed67/68, HPO, persistent mix,
completion, JEPA, other Gate, new Query or Memory write is authorized here.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -m \
  experiments.osram_text_path_router_joint_mosi_20261019.orchestrate \
  --source-commit SOURCE_SHA --output NEW_REMOTE_DIRECTORY --gpu-uuid GPU_UUID
```

Do not run until SSH/resources and the numerical bridge can be verified. A
failed bridge/smoke is not a model-performance result and must stop this queue.
