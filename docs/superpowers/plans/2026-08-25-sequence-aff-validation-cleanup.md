# Sequence AFF validation cleanup plan

## Scope

Clean only the Sequence AFF validation and experiment-support surface. Do not
change the AFF equation, GraphModel behavior, locked masks, runner scheduling,
or any active remote result.

## Behavior lock

The retained checks must prove exactly four contracts:

1. **Method contract (no training):** seven-pattern encoding, masked temporal
   context, complete-modality exact addition including direct gradients, and a
   non-complete content/mask-dependent output.
2. **Integration contract (no training):** the default addition path preserves
   parameters/RNG/output, while `mask_sequence_aff` changes only the branch join.
3. **Archive contract (one epoch, opt-in):** a compact filename is writable and
   full configuration/mask/parameter provenance remains inside the NPZ.
4. **Formal evidence contract (no training):** 8 rates x 5 seeds, inherited
   Original, candidate mask-hash pairing, and paired summary statistics.

## Smells and ordered passes

1. **Boundary violation:** move Sequence-AFF integration assertions out of the
   generic MPFiLM integration test into a dedicated integration test file.
2. **Duplication:** keep complete-modality forward/backward equivalence at the
   module layer and keep only default-path identity at GraphModel layer; delete
   repeated end-to-end forms of the same assertion.
3. **Over-defensive tests:** delete source-text inspection, synthetic non-finite
   bypass, small-channel special case, and exact implementation-shaped parameter
   formula checks. They do not support a paper claim.
4. **Execution boundary:** keep the real one-epoch archive test separate from
   unit tests and document that it runs only after train/save code changes.
5. **Future reuse:** retain the current validated summarizer for this running
   experiment, but forbid copying it for each new module. A later shared result
   reader must be extracted from proven CP-LECC/Sequence-AFF behavior in a
   separate refactor, not during an active formal run.

## Target retained test set

- Method: 5 focused tests.
- Graph integration: 3 focused tests.
- CLI/archive: 2 focused tests (one is opt-in one-epoch).
- Runner: 2 focused tests.
- Summary: 4 focused tests (trusted load, provenance rejection, mask pairing,
  and paired statistics/Markdown).

## Verification

After each deletion/move, run only the affected no-training unit test file.
Run the one-epoch archive test once at the end because archive naming changed.
Do not run a full discovery suite and do not start any smoke or formal job as
part of cleanup.

Commands are deliberately separate:

```bash
# Fast, no-training contract (run after AFF or wiring changes)
python -m unittest \
  tests.test_sequence_aff \
  tests.test_sequence_aff_integration \
  tests.test_sequence_aff_runner \
  tests.test_sequence_aff_summary

# Opt-in archive integration (only after train/save changes)
python -m unittest \
  tests.test_training_protocol.TrainingProtocolTests.test_short_mask_sequence_aff_run_records_full_provenance
```
