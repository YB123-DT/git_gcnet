# Preserve the guard while accepting verified disabled integration

2026-10-21, INTERNAL TEST-ORACLE SCREEN. No new performance result.

First attempt: source `cbf0dea323b2d08b4403b826a3cbc42f11fd6c5b`,
remote `execution_v1`. Both jobs failed before the first optimizer step with
`ValueError('Training config changed beyond backbone learning rate')`.
The failed directory and logs remain untouched;it contains no trained result.

Cause: the reused legacy LR worker required the original reference's exact
config keys and source hashes. Current source has since integrated two
default-disabled fields (`osram_source_role=False`,
`osram_source_role_mode='residual'`) and corresponding gated code in
osram.py,model.py,train_gcnet.py. These were absent in the original Flat
reference metadata. The model was NOT accidentally running source-role.

Resolution is limited to this new entry point: normalize only those disabled
defaults,accept only the explicitly recorded before/after SHA256 pairs,and
preserve the original strict check for every other setting/source. Record
`core_sources_exact=False` and the exact compatibility changes instead of
pretending source bytes are identical. The old LR entry point's guard remains
unchanged unless this dedicated validator is explicitly supplied.

Verification:15 focused CPU tests pass,including unknown source rejection,
enabled-source-role rejection,nonmutating metadata normalization,the150epoch
schedule boundaries,old constant-LR defaults,and exactly two seed66 Flat jobs.
In the archived s0 environment,current disabled OSRAM was compared against
the original27b8c1f source:initialization RNG,train-mode forward/context,
backward gradients,and training RNG all match exactly. The mandatory real
MOSI protected-checkpoint prediction bridge is retained on every actual job.

Restart uses a new immutable source snapshot and independent `execution_v2`;
do not resume the zero-epoch failure,overwrite its logs,or mix source versions
in a performance ranking. Only the originally authorized two jobs are run.
