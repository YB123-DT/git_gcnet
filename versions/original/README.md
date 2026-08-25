# Original

`variant.py` is intentionally a no-op descriptor. The locked configuration
selects PyG `RGCNConv` and direct temporal/speaker addition. It adds no
parameters and consumes no additional initialization RNG.

`reference/` preserves the four official upstream Python files that differ
from the shared, multi-version runtime. They are read-only provenance copies;
training through `run.py --version original` still uses the shared runtime and
the no-op configuration.

Results: [`../../results/iemocap6/fold5/original`](../../results/iemocap6/fold5/original).
