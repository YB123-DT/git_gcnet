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

CPU checks passed: decoder-only initialization difference, unchanged RNG and
other parameters, nonzero initial output, inactive/padding/NaN masks, finite
first-step gradient and actual core update. Trainer import verified.

Status: running, launched 2026-10-08 10:41 UTC. Snapshot commit2d2de17;
tmux nested_direct_random_20261008, dispatcher PID3296507, training PID3296670.
Training log seed_66/train.log confirms loaded feature dimensions512/1024/1024.
DISPATCH.json and PROVENANCE.json track completion. No performance claim yet.

## Completed result

100 epochs completed, exit0 and outputs_verified=true; all recorded artifact
SHA256 hashes rechecked. Existing per-rate BEST retained; no new inference.

|Metric|Eight-rate|High (.5/.6/.7)|
|---|---:|---:|
|ACC|80.221|74.848|
|W-F1|80.177|74.800|

|Rate|ACC|W-F1|
|---|---:|---:|
|0.0|88.110|88.126|
|0.1|86.890|86.877|
|0.2|82.165|82.054|
|0.3|80.793|80.680|
|0.4|79.268|79.278|
|0.5|74.695|74.695|
|0.6|75.000|74.918|
|0.7|74.848|74.787|

Random decoder initialization recovers +2.861910pp mean8 W-F1 and +3.488359pp
high W-F1 over direct-zero. Still -0.815229pp mean8 and -1.277021pp high versus
original residual Nested. Initialization affected this run materially, but does
not explain the entire residual/direct gap. One seed only; no general mechanism
or statistical significance claim.
