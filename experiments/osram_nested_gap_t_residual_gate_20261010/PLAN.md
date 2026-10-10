# Gap-T-only variation of the approved Nested residual Gate

INTERNAL DIAGNOSTIC ONLY

User requests only Gap-T gating. Keep the previous network, dimensions and gate
parameter budget; enable only modality index1 (A/T/V order). Active Gap-A/V retain
coefficient1. Original inactive slots retain0. No parameter freeze, new loss,
Memory/query modification, or staged training. Original Local/Base unchanged.

One new registered method: `nested_gnn_gap_t_residual_gate`.
Test first: exact initialization/RNG parity; same capacity as all-Gap Gate;
nonunit gate changes only T residual; T-present rows cannot receive gate effects
or task gradients to gate parameters; finite updates from T-missing rows.
Run the existing tests and retain existing original/all-Gap method behavior.

Then seal an immutable source commit and launch MOSI seeds66/67/68,100epochs,
eight cyclic rates, identical original cfg84 optimizer/loss/batch and per-rate
Test-oracle selection. Reuse the prior verified runner and TrainingState.
Save all eight BEST files, predictions and full recovery state. No baseline rerun.

Server stays biggpu, physical GPU7 UUID
`GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`; GPU4 forbidden.
New source/logs/results root is
`/data1/yb/remote_experiments/osram_nested_gap_t_residual_gate_20261010`:
179GB free versus19GB on/data2. Dataset and same-seed reference configurations
remain on/data2. Do not remove or relocate previous runs. Admit seeds incrementally
then run concurrently if actual GPU resources permit.
