# Conditional historical-Text path decision Router

INTERNAL FROZEN-ANCHOR EXPERIMENT. User-specified three-candidate probability
mixture, not a new Memory, query, TASR, or readout feature Gate.

## Approved design and finite execution plan

- [x] Inspect current original Flat, source tracker, previous MELD/IEMOCAP audits,
  train sampler and exact selected masks. Preserve all protected artifacts.
- [x] Test-first Router primitive: detached inputs, stable probability mixture,
  exact noneligible bypass, RNG isolation and finite parameter updates.
- [x] Test-first candidate replay and runner: one causal scan, two extra readouts,
  no parameter changes, no gradient/optimizer step on empty eligibility.
- [ ] Real GPU smoke on MELD and IEMOCAP-6, verify cached candidate parity and
  checkpoint/optimizer/RNG restoration before the two finite training jobs.
- [ ] Train seed66 once per dataset, 30 epochs; select one Router on validation,
  replay fixed test masks at all eight rates, summarize and push small artifacts.

Source anchors are exactly the fixed miss0.3 original Flat used by the completed
four-condition audits. MELD seed66 lr1e-5/wd1e-5 anchor epoch21 SHA256
`6f031b515723b441129e16a2e6217410e0eceee2c468e1cbe7b7c3d8276bdb7d`;
IEMOCAP-6 seed66/fold5 anchor epoch69 SHA256
`6a7968ade42513ee868e6dc82f306274a2e7bd53d1943c05f00c1a75af530fd0`.
No per-rate change of backbone. Original anchors were Test-oracle selected;
Router selection does not use test labels.

## Computation and exact trigger

`eligible = valid & current_Text_missing & strictly_past_Text_ever_seen`.

One original causal Memory scan yields Local, Base, active Gap and source traces.
Original input, queries, K/V, joint write, Local Skip, norm/head and task definitions
are unchanged. All backbone parameters are frozen and all backbone dropout is eval.

Candidate order:

1. BG: original full Flat output.
2. B: subtract only Gap-T←T before the same Flat/norm/head.
3. G: subtract only Base←T before the same Flat/norm/head.

Other Gap rows and sources stay present. Both-off is not a fourth candidate.
`p = sum_k softmax(router_features)_k * softmax(candidate_logits_k)` is computed
using logsumexp; do not average logits or candidate CE. Else use original full
log-softmax exactly and fixed weights `[1,0,0]`. No label is a Router input.

Router inputs are actual post-local-path Local256, forward Base←T512,
forward Gap-T←T512, count/coverage/age of strictly past Text, both read norms
and their safe cosine. Three separate Linear64 projections plus six scalars
(log1p count, coverage, log1p nonnegative age, log1p norms, cosine), followed by
LayerNorm→Linear64→GELU→Linear3. No dropout; initialization is RNG-isolated.
Final Linear is zero-initialized, so eligible weights start uniform, NOT baseline
identity. Noneligible behavior is exact baseline from the start.

## Fixed first-round protocol (no search)

- Router-only Adam LR3e-4, wd1e-5, 30 epochs, batch32, seed66, clip1.
- Original GCNet random-mask-v1 host/guest padded sampling, original cyclic rates
  `(epoch + batch_index) % 8`; no persistent view, extra dropout, or extra loss.
- Sample-mean NLL across all valid utterances retains original task normalization;
  noneligible terms are constants. Skip optimizer on no-eligible batches (including
  rate0), but still generate masks and scan to preserve the trajectory sequence.
- Only Router optimizer state changes. Save best_router.pt and last.pt, with
  optimizer/RNG/epoch/config/source identity; original checkpoint is referenced,
  not copied or modified. Last is full Router-training recovery, not backbone
  retraining continuation.
- MELD official train/dev/test; IEMOCAP fold5 fixed10% original-training-conversation
  holdout from existing `setup_case`. The historical frozen backbone saw that
  holdout; this is Router-only validation, not a clean backbone validation claim.
- Validation: one cached GCNet mask bank at eight rates, fixed before training.
  Select by mean8 ACC, ties lower CE; record UA/WF1 too. No per-rate Router best.
- Frozen validation candidates can be cached because the entire backbone is fixed.
  Train masks are newly sampled each epoch by the unchanged original sampler.
  Validation/inference and construction leave training RNG untouched.
- Test once after selection with the previous audits' exact saved common masks;
  require full and both single-removal candidates bit-exact to old caches.
  Report whole-test and eligible-subset ACC/UA/WF1/CE, per-rate/high means,
  corrections/harms, weight distributions and original/constant-uniform controls.

## Limitations

Persistent conditions have no eligible targets: Text is either always present or
never seen, hence this method cannot improve those predictions. This is a known
scope boundary, not an error to repair by changing the trigger or missing sampler.
No oracle routing, threshold search, auxiliary loss, new seeds or other folds.
A convex probability mixture cannot exceed the best candidate's probability for
the true class, though its argmax may differ from all three candidates' argmax.
Implementation success does not imply task gain or publication novelty.
