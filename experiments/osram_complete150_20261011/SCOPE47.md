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

Verified handoff_complete, original six child PIDs still alive, two cancelled
children exited, all15excluded records cancelled_by_user. New eligible6class
jobs fill GPU3 vacancies. No new training protocol or model changes.
