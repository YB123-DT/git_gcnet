# MELD Information Availability Probe Design

## Objective

Locate MELD's weak history benefit without changing or retraining OSRAM:

- **A — weak source history:** strict-past utterances contain little emotion
  information beyond the current Local representation;
- **B — storage loss:** simple strict-past summaries contain information, but
  OSRAM Base/Gap do not preserve comparably decodable information;
- **C — readout under-use:** Base/Gap contain conditional task information, but
  the trained Flat path gains little from it.

This is an internal frozen-representation diagnostic, not a new model result.
P0 path interventions and P1 content interventions are reused as behavioral
evidence and are not rerun.

## Scope

- Dataset: MELD only.
- Frozen backbone: cfg84 no-JEPA causal Flat, backbone seed 66.
- Missing rates: 0.0 through 0.7, using the retained per-rate checkpoints and
  the existing MELD mask protocol.
- Split: fit probes on the existing training split and evaluate on the existing
  test split. No backbone gradient or parameter update.
- Three deterministic probe initializations are allowed because only small
  offline heads are trained; they do not create additional backbone runs.
- Existing Test-oracle checkpoint selection is retained and prominently marked
  `INTERNAL DIAGNOSTIC ONLY`.

## Frozen feature cache

For each missing rate, run each split through the frozen checkpoint once and
cache valid utterances with conversation ID, position, speaker ID, label,
availability and:

- current `Local` (256 dimensions);
- forward `Base` (512 dimensions);
- forward `Gap-A/T/V` (three fixed 512-dimensional slots), hard-masked so only
  the currently missing modality's Gap can be active;
- strict-past mean Local over all earlier utterances;
- strict-past mean Local from the current speaker;
- history-present/count metadata.

No backward zero half is cached. First turns and absent same-speaker histories
use exact zero summaries plus explicit presence/count metadata. All summaries
are strictly causal and exclude the current utterance.

The extraction verifies frozen model hashes, original prediction parity,
padding exclusion, inactive-Gap zeros and exact utterance identity/order.

## Probe families

Every probe uses train-only normalization and the same small architecture
(`LayerNorm -> Linear(hidden=64) -> GELU -> Linear(7)`). It is trained with the
same optimizer, steps and seed. Input widths differ, so claims rely on
dimension-matched donor controls rather than raw parameter-count equality.

Real representations:

1. `Local`: `[L, availability]`.
2. `Base`: `[B, availability, history metadata]`.
3. `Memory`: `[B, masked G_A, masked G_T, masked G_V, availability, history metadata]`.
4. `Local+Base`: `[L, B, availability, history metadata]`.
5. `Local+Memory`: `[L, B, masked gaps, availability, history metadata]`.
6. `RawHistory`: strict-past all-speaker and same-speaker mean Local plus history
   metadata and current availability.
7. `Local+RawHistory`: current Local plus the same raw-history summary.

Controls are constructed within each split without labels by taking a donor
from a different conversation matched on current availability and history
presence. `Base`, `Memory`, `Local+Base`, `Local+Memory`, `RawHistory` and
`Local+RawHistory` each receive an exact-width donor counterpart. Current Local
is never replaced. Unmatched rows are excluded from both real and donor members
of that paired comparison and coverage is reported.

## Metrics and aggregation

Report test ACC and weighted F1 for every rate, eight-rate mean and high-missing
mean (0.5/0.6/0.7), separately for every probe initialization. Report mean and
sample standard deviation across probe initializations. Also report paired
real-minus-donor differences and exact sample coverage. Do not use the probe to
replace the task model or report it as deployable performance.

## Decision rules

- `RawHistory` and `Local+RawHistory` do not beat their donor controls: evidence
  for **A** under this probe class (no readily decodable strict-past signal).
- Raw-history probes beat donors, but Base/Memory probes do not: evidence for
  **B** (loss during learned storage/readout formation).
- Memory and `Local+Memory` beat matched donors, while P0 shows near-zero
  original Flat history gain: evidence for **C** (usable frozen information is
  available but under-used by the trained task path).
- Memory-only above chance but `Local+Memory` not above `Local+donor-Memory`
  means history is decodable yet conditionally redundant with Local; this is
  not classified as C.
- Mixed outcomes remain mixed; no forced single-case conclusion.

The thresholds are descriptive effect sizes with three probe initializations;
this first-pass diagnostic does not claim statistical significance.

## Outputs and stopping rule

Create a new experiment directory and never overwrite P0/P1 artifacts. Save
configuration, source hashes, cache hashes, per-rate predictions, all probe
checkpoints, paired coverage, `SUMMARY.json` and `RESULT.md`.

Stop after MELD seed66 is summarized. Do not automatically expand to additional
backbone seeds, IEMOCAP, a new module or a hyperparameter sweep.

## Verification

Tests must cover strict-past summaries, first-turn zeros, speaker filtering,
inactive Gap masking, split-local donors, label-free matching, paired coverage,
frozen-state preservation, cache identity and metric recomputation. A limited
real-checkpoint smoke must establish extraction parity and finite probe
gradients before the single authorized run.
