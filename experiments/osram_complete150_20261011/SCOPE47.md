# Scope override: no IEMOCAPFour Flat

User explicitly removed all IEMOCAP4Flat runs. Supersedes62-task launch scope;
the original MANIFEST remains immutable for running children's source integrity.
Current QUEUE and CANCEL_FOUR_FLAT.json define47remaining authorized tasks:
MOSI Flat67/68(2),IEMOCAP4Nested(15),IEMOCAP6Flat/Nested(30).

On2026-10-11 old scheduler2149106 was paused and its exact cmdline verified.
Two targeted4Flat training PIDs2149731/2152153 received SIGTERM;13pending4Flat
tasks cancelled. Their logs/recovery artifacts retained, no files deleted.
Old scheduler alone retired; no process-group signal was sent. Six required
running jobs retained their PIDs, model state and logs, without restart.

New persistent controller2167068, tmux socketgcnet_complete150_20261011,
session scope47, controller commitad4555f. Scope controller adopts live jobs
using PID/start-time identities and verifies terminal completion artifacts.
Never dispatches the15excluded task IDs. Model wrapper/source_4eed138 unchanged.
Same resource admission rules and failure handling retained.

Initial handoff_complete observed six survivors briefly, but subsequent checks
found all six interrupted after the old tmux session exited. The earlier
claim of uninterrupted preservation is therefore superseded: scheduler-only
termination was insufficient to isolate child lifetime. Two requested4Flat
children remained cancelled, as did13pending tasks. No result files deleted.

Recovery: pause scope47 controller2167068 (do not terminate its still-live
6class Flat67 child2167167). New controller2177243 adopts that child and
restarts only six interrupted tasks from their own full recovery states.
recover_run.py uses the original manifest and historical sources, restoring
model,Adam,RNG,mask/sampler and selection identities through TrainingState.
Uncommitted history tails may roll back to the last atomic checkpoint.
New children use start_new_session=True, isolating them from tmux teardown.
Original failed logs and queue snapshots remain preserved. No new training
protocol or model changes. Verify recovery advancement before claiming restored.

Recovery verified: all six progressed beyond interruption. MOSI67/68 reached
38/36;FourNested66 reached115;SixFlat66 reached15;SixNested66 reached111;
FourNested67 reached112. Queue8running/39pending/15cancelled/0failed.
The6class Flat67 child2167167 remained uninterrupted. Old controller2167068
intentionally stays paused while that inherited child exists; do not kill its
tmux session before the child finishes. Active coordinator2177243 owns QUEUE.
Recorded recovery controller and wrapper commitdc84f75. Full150 completion
and final metrics are still pending, not implied by recovery verification.
