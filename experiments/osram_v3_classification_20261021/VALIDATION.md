# V3 transfer launch validation

20 new driver/queue tests passed (12 worker,8 queue), including finite job scope,
original configuration guards, same-call ACC selection and matching UA, full
checkpoint state, exact mask-only replay, padding, cohort identity and GPU caps.
Tests were written red before implementation. Compileall and whitespace checks
passed. CPU checks used local Torch2.2.2 only as engineering validation; formal
classification runs require the accepted remote cmerc-repro Torch1.13.1 runtime.

Existing V3 implementation/math/causality/mask/RNG tests and prior MOSI/MOSEI
GPU evidence are retained in osram_source_role_v3_20261021; no model rewrite.
This extension changes no core numerical file.

Remote classification smoke is mandatory before the11 training submissions.
The persistent queue records its measured results under smoke/*/SUMMARY.json:
three real updates, finite/nonzero V3 task-gradient groups, original backbone
and new parameter updates, exact zero-init output, checkpoint reload and
prediction/mask identity. Until those files exist and pass, GPU verification is
pending, not complete. Smoke is not performance evidence.
