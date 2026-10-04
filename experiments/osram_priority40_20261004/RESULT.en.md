# M01–M40 code delivery

INTERNAL DIAGNOSTIC ONLY. All 40 user-specified configurations are implemented; none was trained in this delivery.

37 R variants supplement the original Flat pre-normalization anchor with a zero-initialized residual.
M11/M13 modify only Adapter inputs, retaining the original Local skip. M30 replaces only
`emotion_adapter[0]` with DyT, preserves final `emotion_norm`, and does not claim LN equivalence.
Memory, queries, task head, task loss and missing-mask protocol remain unchanged.
Only the forward 512 history dimensions enter the new operators. Inactive Gap slots and padding are sanitized.

22 earlier source-grounded implementations are also integrated. These are 62 implementation/configuration
entries, **not 62 novel mechanisms**, and the requested 120-candidate expansion is not complete.
Known historical mechanisms are recorded as comparisons, not renamed new methods.

The configuration generator copies the real cfg84 seed66 configuration, changing only
`osram_meaningful_block`. The reference SHA256 is
`65ba11e17dff20264d05f60932c00ae389f6efa7c3f0acf10720db659464a2ba`.
Its per-rate Test-oracle checkpoint protocol remains an internal diagnostic, not validation-selected paper evidence.

CPU contracts, full-model zero-start/RNG checks for the 39 R/I variants, DyT scope, and regression checks
are documented in `VERIFICATION.json`. Parameter counts and generated configurations are in `configs/SUMMARY.json`.
No new GPU smoke or training was launched; a pending PointCNN refill waiter was stopped while existing training was preserved.

These are mechanism adaptations, not original benchmark reproductions. Source links and departures are
recorded in each implementation. Original-author code was not confirmed for ASP/Cross-stitch;
Dynamic ReLU uses a later author implementation for comparison, and TFN a reference implementation.
NODE initializes thresholds from the first valid training input only, never evaluation statistics.
There are no new W-F1 results, checkpoints, GPU-stability conclusions or performance-improvement claims.
