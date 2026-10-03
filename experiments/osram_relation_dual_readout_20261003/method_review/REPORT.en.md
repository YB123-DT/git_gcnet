# Relation training-method review

INTERNAL DIAGNOSTIC ONLY

Date: 2026-10-03. Local evidence revision: d2dc748. This is a literature/code
screening and conditional design, not an implementation or experiment. No model
inference, model-gradient audit, optimization step or parameter search was run.
The literature-trace workflow requires source links and bilingual artifacts;
the design workflow keeps this turn at proposal stage.

## Recommendation and evidence

Keep the current Relation dual-readout architecture. Consider **two-objective
PCGrad on shared parameters**, conditional on a no-update training-gradient audit.
This is an optimization change, not a new forward block, missing-feature generator
or solution to unavailable-modality information. Its standalone architectural
novelty and its OSRAM performance are not established.

The [three-seed result](../MULTISEED.md) is 80.315 eight-rate / 75.366 high-missing,
versus Flat 80.559 / 75.594. The older single-loss Relation seed66
[residual-off audit](../../osram_current_history_relation_20261003/residual_off_analysis/RESULT.md)
gave off-minus-Flat -0.707 and on-minus-off +0.216 points. This motivates separating
branch effects from training changes; it does NOT establish gradient conflict in
the dual model. Existing Gap audits also distinguish retrospective harmful cases
from observables capable of identifying them. All scores and candidate selection
are exploratory Test-oracle evidence, not validation-selected paper results.

Unlike the previous unconditional input-detachment proposal, PCGrad intervenes
only when objective gradients conflict. Neither alternative has been evaluated
here; do not combine them or claim one is empirically superior.

## Screened sources

| Method | Relevant principle | Fit and review scope |
|---|---|---|
| [Side-Tuning](https://arxiv.org/pdf/1912.13503), ECCV 2020 | Fixed pretrained base plus a trainable side network | Conflicts with joint from-scratch training. Read §3.1, §4.4–4.5 and author merge code. |
| [Gradient Similarity](https://arxiv.org/pdf/1812.02224) | Scale auxiliary updates using gradient alignment | Main/auxiliary asymmetry needs care when Full is deployed. Read §2–3 and limitations after Proposition 1; no author implementation verified. |
| [PCGrad](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf), NeurIPS 2020 | Project conflicting objective gradients | Conditional first candidate; read §2.2–2.4, Algorithm 1, §3 and author TF implementation. |
| [CAGrad](https://arxiv.org/pdf/2110.14048), NeurIPS 2021 | Optimize worst local improvement near the mean gradient | Additional c and inner solver; not the first simultaneous change. Read §3.1–3.2 and author toy implementation. |

Author code: [PCGrad_tf.py](https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py),
[CAGrad toy.py](https://github.com/Cranial-XIX/CAGrad/blob/main/toy.py),
[Side-Tuning merge](https://github.com/jozhang97/side-tuning/blob/master/tlkit/models/sidetune_architecture.py).
External commits were not pinned. Side-Tuning's frozen-base requirement is
verified in the paper, not a complete tracing of freezing through its generic
merge implementation. External benchmark results were not reproduced.

## Exact OSRAM mapping (our proposed adaptation)

The **base readout includes Local and all valid Memory evidence**, not just the
Base slot. One causal scan and the same Flat adapter output produce
`base=Head(LN(u))` and `full=Head(LN(u+Relation(L,B,G,a)))`.
Keep both task primitives and their nominal 0.5/0.5 weights, architecture,
random-missing protocol, Adam, clipping and weight decay. No detach, frozen
parameters, additional loss, teacher or query. Gradient manipulation means the
update is no longer ordinary differentiation of the displayed scalar mean loss.

Let theta be shared original trainable parameters and phi the Relation-only
parameters. Compute unweighted task gradients b=grad_theta(L_base),
f=grad_theta(L_full), and d=dot(b,f). If d>=0 use (b+f)/2. Otherwise use
`p_b=b-d/||f||^2*f`, `p_f=f-d/||b||^2*b`, then `(p_b+p_f)/2`.
Both projections must use the ORIGINAL vectors. Always keep
`grad_phi=0.5*grad_phi(L_full)`. Phi must not enter shared dot products.
This is symmetric two-objective projection, not a new asymmetric anchor method.

Implementation implications:

- The author implementation is TensorFlow and sums projected gradients; an
  explicit PyTorch adaptation must preserve our averaging and parameter ordering.
  Different dependencies mean dropping None gradients independently is unsafe.
- Specify one shared parameter vector, not unannounced per-layer projection.
  Preserve None for parameters unused by both losses, so weight decay is not
  newly applied. Handle nonfinite values and near-zero norms explicitly.
- Merge before the existing gradient clipping and optimizer step. At zero
  residual, aligned shared gradients must match ordinary dual supervision.
- One forward does not mean unchanged cost: separate reverse passes and graph
  retention add time and memory. Report measured overhead if implemented.
- Inference remains Full-only, independent of other test samples' modality
  availability. No new history storage, missing-latent write or second trajectory.

## Limits and sole next action

Negative inner products alone are not proof of damaging optimization. These two
nested readouts concern the same task, unlike the independent benchmark tasks in
the cited work. Noise from minibatches, dropout or masks can affect the statistics.
Exact antiparallel gradients can zero the shared update. Euclidean first-order
compatibility is not a guarantee after Adam preconditioning/momentum/decay, nor
does it ensure lower task loss or higher validation/test W-F1. The original Flat
function and learned Memory trajectory are not frozen or mathematically preserved.

The sole proposed NEXT ACTION is an **existing-checkpoint, no-update training-data
gradient audit**. On fixed permitted training conversations/masks, record cosine,
conflict counts/denominators, gradient norm ratios, merged norm/direction changes,
and extreme opposition/zero norms per seed, checkpoint-associated rate and batch.
Use a shared train-mode dropout forward and an isolated loaded model; record RNG
and buffer state, prohibit optimizer.step and check unchanged parameter/checkpoint
hashes. No test labels or retrospective same/opposite groups enter this decision.
Snapshots do not reconstruct the whole training trajectory. Mostly aligned,
sporadic-conflict or update-cancelling results are reasons not to launch PCGrad;
stable conflict only motivates a controlled trial, not a causal conclusion.

No such audit was run in this task. Only source inspection and pure-vector
algebra checks were performed: 1000 random pairs (seed66; 510 conflicting,
490 aligned), unchanged aligned updates, antiparallel cancellation, and a
diagonal-preconditioning counterexample. These are not model correctness tests.
