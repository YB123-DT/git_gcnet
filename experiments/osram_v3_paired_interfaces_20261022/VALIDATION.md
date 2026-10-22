# Paired MOSI interface verification

INTERNAL TEST-ORACLE SCREEN. Starting source c53cae714560f90f6f0dd3598b383c22440bcf9e.

Three hypotheses are implemented as exactly six seed66 runs. Each pair differs
only in its designated connection. Existing CHM, original data, queries, joint
write, causal scan and task head remain unchanged. Local/task variants preserve
raw Base/Gap and original Local for Memory; LocalSkip intentionally receives
updated Local only in local-shared. Task-post-norm returns LN(anchor)+deltaH to
the original single Linear, applying its bias once and sharing its weight.

Default standalone parameter counts: topology335006, Local267678, task613086.
Each A/B pair has identical parameter tensors and RNG after construction. The
topology-full core is exactly equal to original standard V3, including gradients.
The diagonal case retains V/O/residual/Norm and original one-source bypass;
source Q/K softmax derivatives are structurally zero, so equal nominal parameter
counts do not imply identical effective freedom. Inactive queries have a harmless
valid-key fallback before final zeroing to avoid all-masked NaN backward rows.

Focused tests were written before routing existed and failed at the missing
variant; they pass after implementation. Local CPU validation, using existing
multimodalerc310 Torch2.2.2 plus an existing pytest installation appended to the
module search path, required no installation or environment change:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -c '
import sys
sys.path.append("/home/yangbin/miniconda3/envs/msa_extract/lib/python3.10/site-packages")
import pytest
raise SystemExit(pytest.main(["-q",
"tests/test_source_role_v3_paired.py",
"tests/test_source_role_v3_paired_integration.py",
"tests/test_v3_paired_interfaces_runner.py",
"tests/test_source_role_v3.py", "tests/test_source_role_v3_integration.py",
"tests/test_source_role_v3_connections_integration.py",
"tests/test_gcnet_legacy_random_mask.py", "--tb=short"]))'
```

Observed **61 passed in7.74s**. Coverage: exact formulas, same-pair initialization,
zero-start complete predictions, source diagonal isolation/finite gradients,
no-history/ATV/padding/inactive sources, shared-head bias once, first-update
Local-projection gradient difference, four-update upstream task gradients, strict
checkpoint restore, single scan, configuration/factory routing, exact six-job
scope, B-minus-A reports, old baseline/V3 outputs/gradients/RNG and GCNet sampler.
Independent read-only integration review found no formula mismatch; the routing
comment was updated to describe the intentional Local Skip change accurately.
Compilation and git diff --check passed.

Real-data s0 GPU smoke, immutable deployment and formal PID/protocol evidence are
recorded separately in LAUNCH.json after observation. CPU correctness is not a
claim that any of the hypotheses improves performance.

The immutable source commit is 2ff0977. Remote original s0 Python/Torch2.2.2
passed the 27 new standalone/integration/runner checks in8.81s. Dataset deployment
includes the original label-directory link; MOSI pickle SHA256 matches the
accepted manifest (`931ee50f3a77b2eb18786840a880a439f3ae6542bf2dfb5e14ecc9690702af84`).
No dependencies or environment settings were changed. Physical GPU2 andGPU3
were healthy/free at admission; GPU4 is excluded by physical index and UUID.

The synthetic first-update gradient test establishes the specified derivative
paths, not that both paths activate at the first real cyclic-training update.
Actual branch activation follows the unchanged GCNet schedule. Nominal parameter
parity holds within each A/B pair; cross-pair capacities differ, so primary
mechanism comparisons are B minus A rather than winner selection across pairs.

## Actual CUDA and formal launch

All six original-s0 real MOSI smokes completed four optimizer updates and exact
full model/Adam/RNG/prediction reload. All original/new parameter updates, finite
nonzero expected gradient groups, source-read connectivity, initial Flat prediction
parity, reference masks/configuration and pair initialization checks passed.

| A/B pair | New parameters each | Smoke peaks A/B MiB |
|---|---:|---:|
| topology-full / topology-diagonal |335006|3356 /3356|
| local-adapter / local-shared |267678|3318 /3318|
| task-pre-norm / task-post-norm |613086|3308 /3294|

At 2026-10-22T23:46:05Z all six formal processes were alive with command lines
matching the controller, and had passed the four-update gradient gate. Epochs
were10/9,8/6,5/4 for the three pairs respectively. Each last saved training row
matched the original Flat training mask trace, update count and LR, with one CHM
scan per optimizer update. Each pair had identical initial module hash and nominal
parameter count. Exact commands, UUIDs, hashes, PIDs and measured formal peaks are
in LAUNCH.json. These are running experiments, not completed model results.

Persistent tmux socket/session: osram_v3_pairs/mosi_pairs. Remote output:
`/data2/yb/remote_experiments/osram_v3_paired_interfaces_20261022/execution_v1`.
No prior run or protected checkpoint was modified, and no additional jobs were
queued beyond the authorized six.
