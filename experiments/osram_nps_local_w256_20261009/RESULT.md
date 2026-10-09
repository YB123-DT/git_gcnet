# Neural Production Local correction: wider rule MLP

INTERNAL DIAGNOSTIC ONLY

One fixed capacity comparison requested by the user; not a new mechanism.
Only the four shared-over-steps rule MLPs change from 128->96->64 to
128->256->64. Keep 4 rules, 3 steps, 64-d tokens, 8 memory heads,
Local/Base/Gap zero-initialized output bridges and original Local Skip.
Memory, query, writes, task head and loss unchanged.

Module parameters: 191808 -> 315328 (+123520, +64.4%).
These are module counts, not whole-model counts.
Old default remains 96. Existing checkpoints keep their old parameter shapes.
New width is a from-scratch experiment, NOT checkpoint continuation.

MOSI seed66, 100 epochs, original cfg84 no-JEPA cyclic random-missing
protocol, same optimizer/lr/batch, eight per-rate BEST checkpoints.
Test-oracle scores are internal screening, not formal paper results.
No automatic multi-seed expansion. No auxiliary loss or additional view.

Implementation: NeuralProduction(rule_hidden), variant neural_production_local_w256,
existing core20 runner and NPS dispatcher (--method). Existing small run untouched.
Plan: bounded CPU shape/mask/gradient check, isolated git snapshot, GPU admission,
launch one run on biggpu GPU5; verify process and effective configuration.

Root: /data2/yb/remote_experiments/osram_nps_local_w256_20261009
Command (from sealed source):

```text
python -m experiments.osram_nps_local_20261009.dispatch
 --root /data2/yb/remote_experiments/osram_nps_local_w256_20261009
 --method neural_production_local_w256
 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
 --gpu-index 5 --gpu-uuid GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62
```

Verification passed on biggpu CPU: width/parameter count, zero-init output
parity, rule weights update with finite gradients, inactive NaN exclusion,
padding evidence zero; old Local variant RNG/parameter parity test passed.
Local syntax compilation and git diff whitespace checks passed.
Initial check failed as expected because the new variant did not exist.
Status: verified; launch pending. No result yet.
