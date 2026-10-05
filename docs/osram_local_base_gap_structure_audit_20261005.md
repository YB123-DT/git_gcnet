# Local-anchored Base/Gap incremental readout: duplicate-mechanism audit

Scope: the user's approval follows the explicitly proposed NEXT ACTION of a
structure comparison, before implementation/training. No code, inference or
training is added in this task. Inspected HEAD: cb7622a.

## Evidence-grounded comparison

| Structure | Actual computation | Original Flat retained? | Increment space | What has already been tested |
|---|---|---|---|---|
| Current–History Relation | Shared per-slot MLP on projected L/C, product, absolute difference and type; active mean; zero-init output residual | Yes | Feature | Seed66 mean81.068→80.576, high76.352→75.685 |
| Gap-increment Filter | Shared Adapter full-minus-gap-zero increment; scalar modulation conditioned on base anchor, increment and availability | Yes | Feature | Three-seed mean80.559→79.579, high75.594→74.699 |
| Evidence-centered Decision Correction | sL; DB(L,B)-DB(L,0); DG(L,B,G,a)-DG(L,B,0,a); equal task supervision at three exits | No | Task score/logits | Three-seed mean80.559→79.290, high75.594→74.164 |
| Latest proposed direction | Encode L/B relationship, then Gap increment conditional on L/B; project feature residual into retained Flat | Intended yes | Feature | Not implemented or trained; design direction, not established mechanism |

All numeric results are INTERNAL DIAGNOSTIC ONLY, per-rate Test-oracle.
Single-seed and three-seed columns are not interchangeable.

Code evidence:

- `gcnet_missing_m3/osram.py:686–762`: CurrentHistoryRelationBlock. Products and
  absolute differences already explicitly encode current/history relationships;
  this is not just a weighting gate. Base and Gap are currently independent
  slot computations until masked mean aggregation.
- `gcnet_missing_m3/osram.py:765–801` and the Flat integration at1780+: the
  GapIncrementFilter already measures a Gap addition after Local/Base through
  the shared Adapter. It is scalar weighting, not new Gap information.
- `gcnet_missing_m3/decision_correction.py:53–64`: gap_condition=(safe_local,
  safe_base). Conditional Gap increment given Local/Base is already implemented;
  it is incorrect to present this conditioning as previously absent.

Reports inspected:

- `experiments/osram_current_history_relation_20261003/RESULT.md`
- `experiments/osram_current_history_relation_20261003/residual_off_analysis/RESULT.md`
- `experiments/osram_relation_dual_readout_20261003/RESULT.md`
- `experiments/osram_gap_increment_filter_20261003/RESULT.md`
- `experiments/osram_decision_correction_20261003/RESULT.md`

## What would and would not actually differ

The most direct expansion of the latest verbal suggestion is a centered
feature branch, e.g. rB=f(L,B)-f(L,0), rG=j(L,B,G,a)-j(L,B,0,a), followed by
LN(uFlat+W[rB;rG]). This equation is an audit illustration, not an approved
implementation specification.

Compared with Decision Correction, retaining Flat and moving increments to
feature space are real architectural differences. They are not exact numerical
equivalence. But both ingredients already exist in the Relation/Gap-filter
family; their combination is a recombination/control, not a newly identified
primitive or justified new mechanism. Changing width, depth, name or role
embedding does not resolve that issue.

Likewise, using a learned L/B summary as the Gap conditioner instead of the
original L/B concatenation may change inductive bias, but alone does not provide
an independent account of what unavailable computation becomes available.
The existing DG already conditions on both L and B. Do not claim it lacked this.

This comparison does not prove that a recombined architecture cannot improve
W-F1. It establishes that the proposed direction has not passed the user's
distinct-meaningful-Block requirement. Improving a capacity/supervision control
would be a different experimental goal and must be described as such.

## Negative evidence must be interpreted narrowly

Relation-off audit: original Flat81.068, trained Relation residual-off80.361,
residual-on80.576. Conditional direct branch contribution is +.216 points,
but the trained retained path is -.707 vs original. Do not claim that the
relation branch universally harms predictions or that it has identified
conflict. Dual supervision improved to80.649 in seed66, still below Flat.

Decision Correction also changed active capacity and supervision: only3,992,487
active parameters versus13,509,793 in Flat, plus three supervised exits. Its
negative result is not a clean rejection of centered increments in isolation.
Conversely, this confound is not evidence that restoring Flat will improve.

## Decision

Reject this loosely specified direction as a new distinct research Block for
now. No model files changed, no experiment launched, no additional test campaign.
Do not rename or combine failed modules and count the combination as a new
source-grounded mechanism. Revisit only with an explicit computational constraint
that existing Relation, conditional DG and scalar Gap-increment filtering do not
already supply, or with a separately authorized and honestly labeled control.
