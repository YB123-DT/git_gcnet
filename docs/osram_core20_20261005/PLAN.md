# C01–C20 mechanism-preserving implementation

Status: IMPLEMENTATION IN PROGRESS. No C01–C20 training result yet.

This batch is authorized by the user's C01–C20 table on 2026-10-05.
It is NOT the previous M01–M40 readout-operator screen, nor a restart of the
withdrawn survey120 queue. Actual working branch: feature/osram-uniform-forced-text;
starting commit: 073984d. Existing unrelated worktree edits are preserved.

## Scope and comparisons

Retain cfg84 data, causal ordering, observed masks, original optimizer/batch and
selection protocol unless the source mechanism explicitly requires a change.
Record every structural/objective difference, source repository revision, source
file hash, adaptation and unverified limitation. Sources are primary papers and
author implementations; third-party ports must not be labeled author code.

| IDs | Implementation lane | Changes requiring explicit reporting |
| --- | --- | --- |
| C01/C02/C04 | storage | Memory organization, compression auxiliary objective, orthogonal state |
| C05–C08 | dynamics | Persistent transition algorithms; C08 observed-only inner updates |
| C09/C10 | probability | State uncertainty/particles, transition and observation updates |
| C11/C12 | probability | Generative representation loss / probabilistic head and task loss |
| C03/C15/C19 | representation | VQ reconstruction+commitment / FactorCL objectives / gradient variance |
| C13/C14/C16/C17 | inference | Fixed point, ACT cost, latent edges+prediction, prototype classification |
| C18 | main | Author-order global Group DRO probabilities, persistent recovery state |
| C20 | optimization | CAGrad shared-gradient solve; matched multilevel mean-gradient control |

No claim that all methods retain the original Memory, original regression loss,
or original task head. C17 requires a binary classification control on MOSI and
must not be silently compared as an identical regression protocol. C20 needs a
matched multilevel task-only control. Required source objectives are not generic
auxiliary losses arbitrarily shared among all candidates.

## Implementation sequence

1. Independent owned module files, provenance cards and small semantic tests.
2. Main integration via default-off selector; default baseline retains parameter
   creation order and behavior. Alternative scans initialize state per forward,
   read before write, never read future inputs, and skip padding transitions.
3. Verify inactive NaN-safe masking, history causality, finite backward and
   inference without labels. Test algorithm-specific invariants, not only shapes.
4. Verify complete training state for persistent optimizer/objective state.
5. Scoped Lore commits, push github actual branch; no weights or dataset in git.
6. Only after integration passes: immutable remote snapshot on biggpu, health
   whitelist excluding host GPU4, independent outputs and resource admission.

## Result policy

Seed66 screening uses the recorded eight-rate selection protocol and is labeled
INTERNAL DIAGNOSTIC ONLY when selection is Test-oracle. Report every completed,
failed and conditional candidate, not only the maximum. Do not automatically
expand weak single-seed candidates or infer a mechanism from a small gain.

Agents implement bounded files; the leader owns integration, verification and
launch decisions. Existing paused continuous20 goal is not resumed by this plan.
