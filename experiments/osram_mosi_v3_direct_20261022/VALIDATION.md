# Validation before immutable-source launch

TDD red:9 core tests failed on missing mode/config,4 worker tests failed on
missing new interface,2 queue/report tests failed on missing implementation.

After implementation, leader independently ran81 tests successfully in9.52s:
direct core/worker/controller tests, original V3 unit/capacity/integration/runner,
MOSI capacity and three-seed worker/queue,MOSEI capacity and grid worker/queue.

Verified default/legacy RNG and predictions, decoder random-vs-zero init with
same parameter names/count, actual Flat read inputs replace rather than add,
unmodified LocalSkip/raw returned contexts, inactive/padding/ATV/history guards,
emotion ablations, strict causality, finite task gradients and save/reload.
Default capacity worker keeps its original schedule/initial equivalence check;
only this wrapper injects100-epoch schedule metadata and non-identity init check.

Real GPU smoke is pending at this source commit. Completion and measured memory,
time and restoration evidence will be appended after actual remote execution.
No formal score or training success is claimed by these unit tests.

First remote smoke (source b93209e,execution_v1) stopped BEFORE any optimizer
updates: the initialization diagnostic called isfinite on string conversation
IDs. This was a metadata-check bug, not a model nonfinite or mask error. A new
regression reproduced the TypeError; fix checks numeric fields for finiteness
and string metadata for exact equality. Failed output/source are retained.
Retry uses a new immutable source and execution_v2, without modifying live code.

After that correction, the leader reran the full targeted suite:82 passed in9.56s.
Real MOSI GPU smoke on physicalGPU7/source9827f30 then completed successfully:
2epochs/4genuine Adam updates,39.164s,3192.389MiB peak allocated. All V3 gradient
groups (including source Base/Gap inputs and all four decoder/query roles) were
finite/nonzero; one CHM scan per actual optimizer batch; original task parameters
and V3 parameters both updated; model/Adam/RNG and predictions reloaded exactly.
Original data/split/initial backbone and exact test masks matched acceptedFlat66.

The persistent controller then started one fresh formal100-epoch processPID3349620;
actual GPU process query confirmed UUIDGPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
(physical7). At capture time it completed epoch1 without error. Formal performance
is pending; smoke scores do not enter the result table. All failed and successful
smoke outputs/weights remain, and ongoing MOSEI source snapshots were not modified.
