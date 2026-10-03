# Evidence-centered hierarchical decision correction

Scope: record the user's overall architecture, not implementation or a training
launch. Current repository reference: 27689a0. This is a new optional decision
readout, not the previous Gap-increment Filter or a residual on original Flat.

## Fixed architecture

Run the unchanged causal ObservedSetEncoder/OSRAM once to obtain L, B and
fixed-slot masked G=[G_A,G_T,G_V], with availability a. No new memory trajectory,
query, write rule, history cache, gate, completion, JEPA or contrastive objective.
Original Flat remains available as the baseline, but is not in this new output
path. No [L,B,G] -> original 1600-dimensional Flat -> old head in this variant.

```
sL   = H_L(L)
dB   = D_B(L,B) - D_B(L,0)
sLB  = sL + dB
dG   = D_G(L,B,G,a) - D_G(L,B,0,a)
sLBG = sLB + dG
loss = (task(sL,y) + task(sLB,y) + task(sLBG,y)) / 3
```

MOSI uses scalar regression scores and the existing task loss; IEMOCAP uses
class logits and its existing classification loss. No softmax before logit
addition. Each loss uses identical valid-sample masking/reduction. Fixed equal
weights; no sweep. One-stage joint training without frozen stages or detach in
the centered differences. Inference returns sLBG; earlier outputs are diagnostic
exits, not label-selected routing or a prediction ensemble.

Within each difference, both evaluations share parameters. D_B and D_G are
separate modules. Follow the specified conditioning exactly: do not silently
add previous-stage scores as inputs. This is hierarchical evidence accumulation;
the networks are not explicitly conditioned on previous-stage logits.

## Requirements for an implementation

- Use nonlinear correction networks. A purely affine map of concatenated inputs
  would cancel its Local-only term and leave a linear evidence projection,
  without Local/evidence interaction.
- Keep both evaluations of each D deterministic relative to each other. If
  dropout exists, reuse exactly the same draw within the pair. Shared weights
  alone do not ensure cancellation. Avoid branch-dependent running statistics.
- Keep a, L and (for Gap) B identical within each difference; only the designated
  evidence is zeroed. Do not set availability to zero in the counterfactual.
- Zero evidence at the same input boundary in both calls, including any shared
  preprocessing. Never zero one branch after a biased projection but the other
  before it. Prefer causal forward read dimensions; original zero backward
  dimensions contain no extra evidence.
- Safely mask inactive Gap and padding with torch.where before processing;
  retain hard zeroing of the correction at no-history/no-active-Gap positions.
  Empty history is defined from valid-prefix history, not t index alone or a
  test of floating-point read norms. No labels determine any mask.
- If correction final layers are zero-initialized, initialization equals this
  model's Local-only predictor, NOT the original Flat checkpoint. Preserve
  original Flat default-off behavior/checkpoint compatibility separately.

## What centering does and does not establish

For a deterministic shared D, evidence=0 gives exactly zero correction and any
additive term depending only on unchanged conditioning cancels. This is an
architectural dependence constraint, not identifiable causal sentiment evidence.
A network could still use nonzero evidence merely as a switch for a Local-based
function. Centering does not prove useful interactions, monotonic improvement
over stages, detection of conflict, or recovery of a missing modality.

At each exit, compare the same utterances: L -> LB -> LBG W-F1 and corrections /
harms, alongside correction magnitudes. Do not interpret a correction's sign or
size alone as success. No new inference or training is authorized by this note.

## Required correctness checks before any run

1. In eval and train mode, B=0 implies dB=0; G=0 implies dG=0, even after learning.
2. First valid utterance has both corrections zero; padding outputs zero.
3. Poisoned inactive Gap/padding cannot influence any output.
4. One Memory scan; paired correction calls share parameters and stochastic draws.
5. Exactly three original-task losses averaged equally; gradients flow through
   both sides of each difference and all intended backbone parameters.
6. Disabling the new readout preserves the original Flat behavior.

## Repository check at 27689a0

osram.py _scan initializes zero memory (line 1333), performs bias-free reads
(1372–1416), then writes current evidence (1423–1432). Padding cannot write, so
the first valid utterance's raw B/G are zero. Causal backward context is zero
(1513–1520). Use already-ablated emotion_base/emotion_gap (1539–1554), not raw
returned contexts. Original Flat is at 1688–1690 and old task head at
model.py:2029. Three-exit loss can reuse train_gcnet.py _task_loss at 1424–1482
through the task-loss integration at 2221–2241. No code change was made here.

Only overall structure is fixed here. Hidden widths, config/API wiring, parameter
budget and experiment launch belong to the next explicitly requested implementation
step; this document does not authorize a new multi-seed run.

## Subsequent execution authorization

The user subsequently requested “开始跑”. The implementation plan dated
2026-10-03 fixes 128-hidden-unit deterministic MLPs, zero-initialized correction
outputs, and the existing MOSI seeds66/67/68 100-epoch screening protocol. That
later request authorizes implementation/launch; the architecture-only scope
above records the earlier design discussion, not a prohibition on this run.
