# Status: corrected preflight, formal training not started

Approved selection correction: MOSI six jobs only, same GCNet masks, restore
per-rate Test-best W-F1. Old validation-selected results remain untouched.
No scores claimed before actual completion; no promise of recovering80.747.

The first remote smoke (`execution_v1`, source420ec79) was stopped by strict
configuration comparison before training: JSON list versus runtime tuple for
the identical eight missing rates. Canonical JSON comparison fixes container
representation only; a regression test still rejects different numeric rates.
The failed source, logs and output remain intact. Corrected deployment must use
a fresh immutable source directory and `execution_v2`.

Verification:50 related tests passed, including the container regression and
unchanged GCNet legacy random sampler checks. GPU smoke remains pending.
