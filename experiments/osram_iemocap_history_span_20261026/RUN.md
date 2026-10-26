# Executed frozen inference

- Status: complete, 2026-10-26, ssh biggpu.
- Audit code: `bf77e0d47ca0f772cb0300705b9f0ebe4f968975`.
- Numeric source: `21e46f2ecd092345818659694db8ff372672c35e`.
- Immutable overlay: `/data2/yb/remote_experiments/osram_iemocap_history_span_20261026/source_bf77e0d`.
- Outputs: `/data2/yb/remote_experiments/osram_iemocap_history_span_20261026/execution_v1`.
- Log: `/data2/yb/remote_experiments/osram_iemocap_history_span_20261026/run_v1.log`.
- Persistent session: `history_span_20261026`; exited after successful completion.
- GPU: physical0, `GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`, logical cuda:0.
- Runtime: `/data2/yb/conda_envs/cmerc-repro/bin/python`, Torch1.13.1+cu117,
  NumPy1.26.4, sklearn1.3.2. Exact original environment bridge is recorded in JSON.
- Total wall:44.90sec, peak allocated706.50MiB. No optimizer or training.

```sh
cd /data2/yb/remote_experiments/osram_signature_iemocap_cyclic_20261025/source_21e46f2
PYTHONPATH=/data2/yb/remote_experiments/osram_iemocap_history_span_20261026/source_bf77e0d:$PWD \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_iemocap_history_span_20261026.run \
 --source-commit bf77e0d \
 --output /data2/yb/remote_experiments/osram_iemocap_history_span_20261026/execution_v1
```

The output is intentionally non-overwritable. For a new replay, change only its
output directory, not protected parent paths. No running task was interrupted.

## Verification

Five CPU tests passed locally and in cmerc-repro (1.78sec/2.32sec). They compare
window reconstruction against independently truncated original `_scan` calls,
including interior padding, conversation isolation and future edits.

Real full-history replay: maximum absolute error is **0** for Base, Gap and
logits; relative logit error0; argmax exact. Original miss0.3 logits reproduce
the protected selected artifact bit-for-bit. Eight original scans, unchanged
replay CPU/CUDA RNG, all model parameters frozen/without gradients and unchanged
backbone digest. Saved masks preserve their original SHA256. Raw NPZ hashes,
sample alignment and128 all/common metric groups were independently recomputed.

`compileall` and `git diff --check` passed. There is no configured standalone
typechecker/linter added for this inference-only package. The historical optional
torch-sparse library warning remains; this native tensor Flat audit does not use it.

## Narrow interpretation

On the eight-rate average, capping history at8 rather than full changes ACC
62.616→60.798 and UA62.092→59.078. At16 turns ACC is62.223; at32 it is62.477.
The same651 targets with at least32 prior turns show the same broad trend:
ACC58.948/60.945/61.751/62.097 for8/16/32/full. Thus choosing8 as an arbitrary
capacity boundary is not supported here. Most of the measured task benefit is
approached by16–32 turns, with a remaining full-history advantage on average.

Not every individual rate is monotonic: at0.7, K32 ACC57.425 exceeds full56.932,
while CE1.15230 is worse than full1.14548. This does not authorize choosing K32
from test or claiming uniformly better truncation. See complete per-rate and
current-pattern results in CSV/JSON, including corrections and harms.

This is one original checkpoint, one seed, one LOSO fold and a potentially
out-of-distribution state intervention. It tests task dependence on retained
history extent, not a hard mathematical storage limit or semantic forgetting time.
