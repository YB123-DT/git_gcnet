# MELD Flat history-scale diagnostic

**INTERNAL DIAGNOSTIC ONLY.** Frozen inference on the retained causal MELD
Flat checkpoints. One unchanged causal scan is reused while Base and all
already-masked Gap inputs are multiplied by `alpha` in `{0, .5, 1, 1.5, 2}`
immediately before the original Flat adapter. Local, classifier, masks and
weights remain fixed. `alpha=1` must exactly reproduce the original model.
