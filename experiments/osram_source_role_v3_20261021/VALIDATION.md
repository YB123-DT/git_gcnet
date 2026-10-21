# V3 essential validation

INTERNAL TEST-ORACLE SCREEN

2026-10-21: s0 Python, remote CPU, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`:

```bash
OSRAM_V3_LEGACY_SOURCE=/path/to/exported/7d9e9c4/gcnet_missing_m3/osram.py \
  python -m pytest -q tests/test_source_role_v3.py \
  tests/test_source_role_v3_integration.py --disable-warnings --tb=short
```

Result: **16 passed**, one warning, 6.81s. Legacy source was exported directly
from committed `7d9e9c491b11dee09d3f8560c0b111a911df5818`.

Covered: exact343 pair input; individual Gap messages before aggregation;
shared read coordinates; role-isolated attention; zero/one/multiple-source
handling; inactive/padding/ATV/no-history safety; finite gradient paths; default
off legacy output/state/RNG/gradient equality; zero-init predictions with a
nonzero original Flat adapter; single scan; causal independence; original
forward read-slot placement; checkpoint restoration; CLI/config/model factory.

The original Flat adapter and V3 decoders both initialize to zero. Upstream V3
gradient checks must allow real optimizer updates to open both layers; a zero
initial upstream gradient alone is not a wiring failure. Tests explicitly
distinguish initial identity from later gradient connectivity.

Default V3 parameter count: **335,006** added; expected total13,844,799 with
the accepted cfg84 original13,509,793. No new task head or loss.

GPU smoke and actual launch evidence are recorded in `LAUNCH.json` and
`RESULT.md`: both real-data smokes passed, zero-init predictions and restored
predictions exact, all41 monitored gradients finite/nonzero. Peak allocated
memory3198.5625MiB(MOSI)/2244.94384765625MiB(MOSEI). Passing tests/smoke does
not imply improved sentiment performance.

Standalone runner tests: **6 passed**, 0.18s on the local CPU. The original
`block_write`, `_scan`, `_address_residual`, `_project_sequence` ASTs match
`7d9e9c4` exactly. Smoke uses4 actual MOSI updates and3 MOSEI updates; this
allows both zero-initialized layers to open without artificial initialization.
