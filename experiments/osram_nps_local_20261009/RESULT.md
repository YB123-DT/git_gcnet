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
