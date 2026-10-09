# Neural Production: add Local adapter correction

INTERNAL DIAGNOSTIC ONLY

MOSI seed66, 100epochs, original cfg84 no-JEPA task loss, cyclic random masks,
per-rate BEST. One new run. Four rule MLPs, three steps and all selection
mechanisms unchanged. Existing Base/Gap residual decoding unchanged.

Add zero-initialized64->256 bridge from final internal Local token:
L_adapter = L + bridge(final_Local_token). Original Local Skip remains L.
No Memory write/read/query changes, extra loss, additional view or completion.
Reuse one existing rule-execution pass. New bridge initialization isolated
with fork_rng; all old parameters and downstream RNG remain matched.
Local delta can be nonzero even if Local was not selected internally: bridge
decodes the final token, consistent with existing Memory decoding semantics.

References: old Neural Production seed66 ACC81.021341, W-F180.981287,
high W-F176.475237. Three-seed old W-F180.458846/high75.557937.
Server biggpu GPU6 only. Root:
/data2/yb/remote_experiments/osram_nps_local_20261009
Command: python -m experiments.osram_nps_local_20261009.dispatch --root RUN_ROOT
--data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
Verification passed remotely: old175168 parameters, new191808 (+16640).
All common parameters and RNG match; zero-init outputs exactly match old NPS;
Local bridge learns under task-like gradient, gradients finite and masks safe.
Trainer import checked. Sealed source6b33729 deployed and pushed.

Status: QUEUED, not training yet. Persistent tmux nps_local_20261009 and
dispatcher PID3184101 verified alive. GPU6 free4499MiB, below the unchanged
8048MiB admission threshold. Dispatcher automatically starts this single
seed66/100epoch run when GPU6 and disk admission pass. No duplicate launch,
no unrelated task stopped. Live status: remote DISPATCH.json; dispatcher.log.

## Authorized alternate GPU launch

User subsequently authorized other GPUs. Old pending dispatcher3184101 was
stopped only after verifying status=pending and no child processes. Old
DISPATCH.json retained as cancelled_before_training; no training was killed.

Current root: /data2/yb/remote_experiments/osram_nps_local_20261009/alternate_gpu5
GPU5 UUID GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62, free13603MiB at launch.
Same server/config/seed/data; module/trainer unchanged. Snapshot3f47531;
tmux nps_local_gpu5_20261009; dispatcher3198586, training3198751.
Started2026-10-09 02:25 UTC. Status RUNNING supersedes GPU6 queue above.
Launch adds --gpu-index5 --gpu-uuid GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62.
GPU4 remains forbidden. Existing GPU5 jobs untouched; timing is shared-load.

## Completed seed66 result (2026-10-09)

Status COMPLETE supersedes RUNNING above: 100 epochs, exit_code=0,
outputs_verified=true; all 20 recorded artifact SHA256 hashes rechecked.
Eight BEST checkpoints and predictions plus last_training.pt retained remotely.
Equal mean across rates; ACC uses each W-F1-selected epoch, not separate ACC selection.

| Missing rate | BEST epoch | ACC (%) | W-F1 (%) |
|---|---:|---:|---:|
| 0.0 | 57 | 87.652439 | 87.681724 |
| 0.1 | 56 | 85.518293 | 85.491887 |
| 0.2 | 78 | 83.079268 | 82.997018 |
| 0.3 | 97 | 80.030488 | 80.125701 |
| 0.4 | 72 | 80.182927 | 80.303357 |
| 0.5 | 67 | 77.134146 | 76.837980 |
| 0.6 | 44 | 75.914634 | 75.031773 |
| 0.7 | 90 | 75.000000 | 75.072884 |

8-rate ACC 80.564024; W-F1 80.442790; high-missing W-F1 75.647545.
Compared with old NPS (Base/Gap correction only), W-F1 changes:
8-rate -0.538497 pp; high missing -0.827692 pp. This seed does not support
adding Local correction. No causal mechanism inferred from this one run.
INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle, not a formal paper result.
