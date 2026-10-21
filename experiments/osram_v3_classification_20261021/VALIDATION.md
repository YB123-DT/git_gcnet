# V3 transfer launch validation

20 new driver/queue tests passed (12 worker,8 queue), including finite job scope,
original configuration guards, same-call ACC selection and matching UA, full
checkpoint state, exact mask-only replay, padding, cohort identity and GPU caps.
Tests were written red before implementation. Compileall and whitespace checks
passed. CPU checks used local Torch2.2.2 only as engineering validation; formal
classification runs require the accepted remote cmerc-repro Torch1.13.1 runtime.

The same20 tests also passed on biggpu in cmerc-repro (5.11seconds). One existing
torch-sparse optional-library import warning was reported; no test failed.
The deployment verified all9654 tracked Git blobs against source commit
`1e44ef3ce36feab63de1e57b7175a1589688c7a2` before launching the persistent queue.

Existing V3 implementation/math/causality/mask/RNG tests and prior MOSI/MOSEI
GPU evidence are retained in osram_source_role_v3_20261021; no model rewrite.
This extension changes no core numerical file.

Remote classification smoke is mandatory before the11 training submissions.
The persistent queue records its measured results under smoke/*/SUMMARY.json:
three real updates, finite/nonzero V3 task-gradient groups, original backbone
and new parameter updates, exact zero-init output, checkpoint reload and
prediction/mask identity. Until those files exist and pass, GPU verification is
pending, not complete. Smoke is not performance evidence.

## Actual remote smoke

All three completed successfully in the original cmerc-repro environment;
exact zero-init prediction parity, finite/nonzero V3 task gradients, backbone
and V3 updates, full checkpoint restoration and selected-artifact replay passed.
See LAUNCH.json for runtime/PID/UUID and measured records.

| Dataset | real updates | peak allocated MiB | elapsed seconds |
|---|---:|---:|---:|
| MELD |3|1650.615|42.167|
| IEMOCAP-4 |3|5349.033|61.092|
| IEMOCAP-6 |3|7222.075|see LAUNCH.json|

Classification smoke is not a50/100epoch run or a performance result. The
larger measured IEMOCAP peak limits actual concurrency below requested4/8 caps
when other tasks and safety headroom consume the remaining memory.
