# P1 results

Status: implementation only; no formal frozen checkpoint run has been launched
by this implementation task. No training or model modification was performed.

Implementation verification (2026-10-14, `multimodalerc310`, CPU): 19 P1
unittests passed, including a real small causal `MissingM3GraphModel` frozen
forward, exact model-state preservation, matched replacements and Local checks.
The existing 12 P0 unittests also passed. Python compilation and
`git diff --check` passed. Real source-checkpoint/GPU smoke, full dispatch,
GPU peak memory, and throughput have not been measured.

Populate this record after dispatch with source/checkpoint hashes, rates,
completed/failed/missing sources, exact/tolerance Local checks, modality coverage,
lag mismatch, paired CE/probability change and corrections/harms. Report every
planned source; retain the internal Test-oracle diagnostic limitation.
