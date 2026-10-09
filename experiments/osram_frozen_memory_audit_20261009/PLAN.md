# Frozen Memory evidence audit

INTERNAL DIAGNOSTIC ONLY

User approved three linked diagnostics; latest override explicitly removes validation and requests direct test evaluation. No backbone training, architecture change, or new main-model checkpoints. Existing OSRAM original LARGE Flat seed66, eight inherited per-rate Test-oracle checkpoints. Fit probe weights on train only; use test MSE for probe epoch selection, clearly not independent generalization evidence. Do not fold validation conversations into train.

## Locked design

1. Capture original adapter inputs via existing frozen-history hook: Local256; Base and three masked Gap forward512 each; availability, labels, speaker IDs, conversation/index, original prediction. `eval`, no gradients, state/checkpoint hashes before and after. Verify test predictions against original saved metrics and check first-turn/inactive Gap zeros. Use original fixed evaluation masks at epoch0; training features use train masks at epoch0, fixed across probe epochs. Store feature caches, not in Git.
2. Offline probes A Local+mask, B Local+Memory+mask, C Local+train-donor Memory+mask. Fixed nonlearned shared512->64 projection for each history slot, preserving Local256. A259->128->1 and B/C515->64->1 have approximately equal trained parameter budgets. Report the exact budgets and unequal widths: this is an approximate capacity control, not proof of information-theoretic uniqueness. Three probe initializations66/67/68, one frozen source seed66,100epochs, Adam .001/weightdecay1e-5/batch128. C donors are train-only, same current availability/history-present, different conversation, chosen without labels; report coverage and match rows across A/B/C. Keep first turn memory zero. Feature scaling uses train data only.
3. Offline targets: current sentiment, immediately previous utterance score, previous up-to-three utterance mean, current-minus-previous score. Labels only targets, not inputs or intervention choices. No jumping over missing indices. Decodable score differences do not establish genuine emotion shift. Same/other speaker analysis is unavailable when conversations contain only one speaker.
4. Model-behavior intervention on test only: original vs exactly one prior observed bit deleted vs exactly one matched control bit deleted. Never delete the last observed modality of a historical utterance; never alter current/future inputs. Compare T vs A/V at same or closest historical location, and nearest vs next-earlier eligible location within A/T/V; optionally same vs other speaker where real support exists. Record lag mismatch, exact deletion provenance and eligibility. Report exact-lag matched modality subset separately, so weak temporal matching cannot masquerade as a modality effect.
5. Summarize W-F1/ACC (nonzero MOSI labels, prediction>0), MSE (all target labels), corrections/harms and counts on the SAME eligible targets. Macro-average per rate, do not pool duplicated test utterances across rates as independent samples. Retain all probe seeds and failed/unavailable cases. Add conversation-clustered uncertainty for paired intervention loss where feasible; no automatic causal or mechanism claim.

## Files and execution

- `probe.py`, `tests/test_frozen_memory_probe.py`: offline normalization, controls, targets and heads (bounded agent).
- `intervention.py`, `tests/test_frozen_memory_intervention.py`: paired masks, captured predictions and isolation checks (bounded agent).
- `run.py`, `tests/test_frozen_memory_runner.py`: extraction, configuration/hashes, GPU whitelist, persistent run records (leader).
- `analyze.py`: summaries and RESULT.md (leader).

Checks precede implementation for new extraction schema; agent tests likewise use TDD. Then one real-checkpoint limited run on healthy biggpuGPU2 checks baseline prediction parity and frozen state, followed by all8rates split into two disjoint shards on healthyGPUs2/3. No GPU4. Independent immutable snapshot, no duplicate starts, exact command/log/PID recorded. Use existing environment, no new dependencies. Save weights locally on server, commit only code and summaries; push current branch to github.

Acceptance: exact IDs and masks, unchanged model hashes, active Gap safety, original prediction parity, no test-gradient fitting, test selection explicitly labelled, donor constraints and matched intervention counts verified. A failed parity/freeze check stops the run instead of producing results.
