# Actual launch,not completed performance results

The finite seed68 NoArea training queue was launched on biggpu at
2026-10-25T14:14:52Z. LAUNCH.json records the actual command,GPUUUID/resource
checks and every overlay source hash. Numerical source21e46f2 stays immutable;
the separately deployed training-control source is
`1d5ea017cb2d99630134722835978c91600e0946`.

Persistent tmux: `signature_noarea_iemocap68_20261025`.
Remote root:
`/data2/yb/remote_experiments/osram_signature_noarea_iemocap68_20261025/execution_v1`.
Controller log: sibling `controller_v1.log`.

The real-data fold5,rate0.7 two-update smoke passed,including original parent
bit-exact initialization,frozen backbone,finite gradients,selected checkpoint
reload,and same-RNG original-control mask/update-trace comparison. Local and
remote24-test suites passed. CPU-backend test limitation is documented in README.
SMOKE.json is implementation evidence only,not a final100-epoch performance.

At14:15:33Z,the queue was in formal training:4running,36pending,no failed jobs.
The first two jobs had reached epoch2. Automatic admission increases up to
6workers on each of physicalGPU0/1/2,subject to resource reserves. No GPU3/4/5/6/7
is used and no unrelated task was signalled. No normal Signature run is retrained.

This is40new100-epoch branch-training jobs,not the earlier roughly100-second
frozen intervention. The final ACC/UA comparison will be generated automatically
only after complete checkpoint/mask/paired-trace validation; it is not available
at launch. Current QUEUE.json and per-worker SUMMARY.json are authoritative.
