# Status: six formal runs and paired reporting complete

Approved selection correction: MOSI six jobs only, same GCNet masks, restore
per-rate Test-best W-F1. Old validation-selected results remain untouched.
No scores claimed before actual completion; no promise of recovering80.747.

Final2026-10-20:all6runs complete150epochs,queuecomplete,zero failures.
Three-seed mean8WF1 Flat78.890919,Router78.936003(+0.045084pp).
HighWF1 Flat72.907882,Router72.624448(-0.283434pp).
Independently rehashed48savedprediction files and recomputed ACC/WF1/MSE:
maximumabsolute discrepancy0.0. RESULT.md/METRICS.csv fetched and sourcehashes
recorded in FINAL_VERIFICATION.json. No new inference or training for analysis.

## Historical launch record (preserved)

The first remote smoke (`execution_v1`, source420ec79) was stopped by strict
configuration comparison before training: JSON list versus runtime tuple for
the identical eight missing rates. Canonical JSON comparison fixes container
representation only; a regression test still rejects different numeric rates.
The failed source, logs and output remain intact. Corrected deployment must use
a fresh immutable source directory and `execution_v2`.

Verification:50 related tests passed, including the container regression and
unchanged GCNet legacy random sampler checks. Corrected GPU smoke completed:
full checkpoint reload and exact selected prediction reload passed;
peak allocation3092.058MiB. This is correctness evidence, not a model score.

Remote source:fd96f540e807426d6b4d858febdc2715b3a79a68.
Archive SHA256:90b7d1180db90ab1c6e2e8958fda3d8683d18823d321ab1f66bbd18ebbcbcb9d.
Immutable source directory:
`/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/source_fd96f54`.
Output:`/data2/yb/remote_experiments/osram_mosi_gcnet_testbest_20261020/execution_v2`.
tmux:`mosi_testbest_fd96f54`; controller PID745063.
Observed six workers:seed66 Flat745710/Joint745732;
seed67 Flat745830/Joint745986; seed68 Flat746075/Joint746186.
Physical GPU7/1/3, two workers each, UUID-bound; GPU4 excluded.
At launch inspection, seed66/67 effective configs explicitly confirmed
GCNet random-mask-v1,150epochs,Test-oracle-per-rate,weighted_f1;
seed68 processes were still initializing. Full training parity and performance
remain pending, not claimed by a launch record. No IEMOCAP/MELD reruns added.
