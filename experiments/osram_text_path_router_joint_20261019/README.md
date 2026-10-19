# One-stage conditional historical Text-path Router

INTERNAL SINGLE-SEED EXPERIMENT. User approval: “行做一次这个一阶段看看咋样”.

## Locked design

Train a fresh original cfg84 no-JEPA Flat backbone and the probability Router
jointly, rather than unfreezing the protected pretrained backbone. Run exactly
four jobs: MELD Flat/control and Joint/Router, IEMOCAP-6 fold5 Flat/control and
Joint/Router, all seed66. No additional seeds, folds or parameter search.

| Dataset | Epochs | Original backbone Adam LR | Router Adam LR | Weight decay | Batch |
|---|---:|---:|---:|---:|---:|
| MELD | 50 | 1e-5 | 3e-4 | 1e-5 | 32 |
| IEMOCAP-6 fold5 | 100 | 1e-3 | 3e-4 | 1e-5 | 32 |

Original dropout0.5, gradient clipping and constant backbone LR are retained.
Each dataset's two jobs use identical original-network initialization, split,
training sampler seed, cyclic rate schedule and fixed evaluation mask bank.
The Router constructor does not consume the original RNG stream. All original
trainable parameters remain trainable, including on batches with no eligible
Router sample. The separate Router LR is a fixed new setting, not an optimum.

For valid utterances with current Text missing and strictly-past Text observed:

1. Scan the original joint-write CHM exactly once, retaining graph-connected
   Local, Base/Gap, joint-Value source reads and original Flat input.
2. Produce three candidates with the same original Flat/norm/head: full BG;
   remove only Gap-T←T (B retained); remove only Base←T (G retained).
3. Reuse the original Flat Dropout realization in both extra readouts; restore
   the RNG state after the original full forward, so subsequent original mask
   and Dropout draws are not shifted.
4. Router features are Local, Base←T, Gap-T←T and strictly-past Text trajectory/
   norm/cosine statistics, never labels. Its softmax weights mix the three
   candidate **probabilities**, using stable logsumexp.
5. Optimize only mean task NLL over **all valid utterances**. Outside the trigger,
   use the exact full candidate log probabilities and still train the backbone.

Uniform initial Router weights are NOT baseline identity on eligible samples.
The main Memory, original Query/K/V, one joint block write, read-before-write,
Local Skip and task head remain unchanged. Source tracking labels joint Value
slot contributions, not pure modality semantics. No new write/query, Gate,
JEPA, prediction, auxiliary objective, persistent augmentation or second view.

## Validation and comparison boundaries

Use MELD official train/dev/test and the existing reproducible conversation-level
IEMOCAP fold5 training holdout. Unlike the older frozen pilot, the newly random
backbone has not trained on the held-out conversations. The matched fresh Flat
uses the same holdout; it is not a rerun of the old all-training-conversation
per-rate Test-best baseline.

Select ONE checkpoint per job on mean eight-rate validation ACC, ties lower CE.
Replay the protected original exact test masks with identities/speakers/labels
checked. Do not select per-rate test epochs or compare new logits to the old
anchor as an identity requirement. Historical Test-oracle mask provenance and
long-running test-guided research make this exploratory, not independent paper
generalization evidence. Protected checkpoints are verified during data setup
then discarded; they are never the training initialization.

Report matched Flat vs Joint overall/high/per-rate ACC, UA, W-F1 and CE; within
Joint report full candidate vs mixture, corrections/harms and Router weights.
A separately trained Flat is not the same model as the Joint full candidate.
Fully persistent missing Text cannot activate this Router; do not infer broad
persistent improvement from the proposed trigger.

## Implementation and verification sequence

- Tests first: differentiable Base source tracking; backward-compatible detached
  default; one scan; shared Dropout; original RNG; all-valid task gradients;
  source/Encoder/Flat/head/Router gradients; recovery and strict masks.
- Remote real-data smoke for each dataset: three training batches, finite gradients
  and changed original/new parameters, limited fixed-mask replay, full recovery.
- Freeze and hash source commit before GPU runs; use the same snapshot throughout.
- Train four finite jobs only after smoke succeeds; separate last/best checkpoints
  preserve full model, optimizer, RNG, config, epoch, split/mask identities.
- Report failed/pending runs honestly. Do not automatically expand failed pilots.

## Commands

From an immutable source snapshot on `ssh biggpu`, using
`/data2/yb/conda_envs/cmerc-repro/bin/python`:

```bash
python -m experiments.osram_text_path_router_joint_20261019.run \
  --phase smoke --mode joint --dataset MELD --gpu-uuid GPU_UUID \
  --source-commit SOURCE_SHA --output NEW_SMOKE_DIRECTORY
python -m experiments.osram_text_path_router_joint_20261019.run \
  --phase train --mode joint --dataset MELD --gpu-uuid GPU_UUID \
  --source-commit SOURCE_SHA --output NEW_RUN_DIRECTORY
```

Repeat `train` for `--mode flat`, and both modes for `--dataset IEMOCAPSix`.
Use `--resume` only for the same run with its complete `last.pt` and identical
source/config identity. Select healthy GPUs by UUID; biggpu physical GPU4 is
forbidden. Never upload weights, feature caches or large predictions to Git.
