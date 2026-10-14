# MELD Flat width screen

INTERNAL DIAGNOSTIC ONLY. RUNNING widths512/1024/2048, baseline1600 reused.
Only adapter hidden width changes. All runs keep `gcnet-random-mask-v1`,
seed66, 50 epochs, batch32, Adam lr1e-3, causal no-JEPA Flat and Local skip.
Per-rate Test-oracle selection is internal screening; no performance claim yet.

All three runs started concurrently on biggpu physical GPU0, UUID
`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`. Historical training code is frozen
at `b4a3138`, runner code at `dc78218`. Config checks confirm that the only
changed field is `osram_adapter_hidden_dim`; missing protocol and rates are
unchanged. Two config regression tests passed. Startup verified epoch001
completed for each run at `2026-10-14T11:36:13Z`.

| Adapter hidden width | PID | Output directory |
|---:|---:|---|
| 512 | 77766 | `/data2/yb/remote_experiments/osram_meld_flat_width_20261014/width512` |
| 1024 | 77767 | `/data2/yb/remote_experiments/osram_meld_flat_width_20261014/width1024` |
| 2048 | 77768 | `/data2/yb/remote_experiments/osram_meld_flat_width_20261014/width2048` |

Logs are sibling `width512.log`, `width1024.log`, `width2048.log` files.
Each run preserves all eight best checkpoint files. Final saved mask/label
parity against the 1600 reference is required before paired UA comparison.
