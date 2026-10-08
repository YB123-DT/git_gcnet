# MOSI direct Nested: random decoder initialization

INTERNAL DIAGNOSTIC ONLY

One new seed66, 100 epochs, existing cyclic random-missing 0.0–0.7 protocol,
same per-rate BEST selection and original task loss. Report ACC and W-F1.
Only change versus direct-zero: local decoder and eight shared-per-head memory
decoders use standard PyTorch nn.Linear initialization (weight AND bias),
instead of zeroing them after construction. No extra RNG draws; core,
tokenizer, downstream initialization and parameter count remain matched.
No output evidence additions; original Local Skip remains intact.

References: residual Nested W-F1 mean8/high 80.992147/76.077229;
direct-zero 77.315007/71.311849; Flat 81.068095/76.352251.
Zero-initialized direct source:
/data2/yb/remote_experiments/osram_nested_direct_20261008/attempt2/seed_66.

Server biggpu GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153.
Run root /data2/yb/remote_experiments/osram_nested_direct_random_20261008.
Command: python -m experiments.osram_nested_direct_20261008.dispatch
--method nested_gnn_direct_random_evidence --root RUN_ROOT
--data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json

Status: pending remote checks and launch; no performance claim.
