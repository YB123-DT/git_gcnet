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
