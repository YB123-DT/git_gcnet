# Memory-only Adapter results

INTERNAL TEST-ORACLE DIAGNOSTIC, not validation-selected paper results.
Code b51b927; launch 56b27e0; biggpu GPU6. All three seeds completed
100 epochs with exit code0; 24 per-rate best checkpoints retained remotely.
Reference Flat reused. Same cfg84 hyperparameters/random-missing protocol;
only remove Local from Adapter input, preserving full Local Skip including bias.

W-F1 percent; differences in percentage points. Each seed averages its eight
per-rate best scores, then aggregate averages seeds. High = rates.5/.6/.7.

|Seed|Flat8-rate|Memory-only8-rate|Delta|Flat high|Memory-only high|Delta|
|---|---:|---:|---:|---:|---:|---:|
|66|81.068|80.012|-1.056|76.352|74.790|-1.562|
|67|80.556|79.498|-1.058|75.990|74.921|-1.069|
|68|80.053|79.525|-0.528|74.440|74.088|-0.352|
|Mean|80.559|79.679|-0.881|75.594|74.600|-0.994|

Seed sample SD: Flat8 .507, variant8 .289; Flat high1.016, variant high.448.
All seeds decline on both aggregates. This tested structural separation does
not improve performance; it does not prove why. Removing Local also shrinks
Adapter normalization/input projection, so this is not a parameter-count matched
causal isolation of Local information. No significance claim from three seeds.
No conclusions about fixed persistent-modality patterns from random-rate scores.
MSE is not used to accept/reject the model; original training loss unchanged.

Raw SUMMARY, configs, metrics, provenance and process completion records are
archived under results/. No checkpoint or sample-level predictions uploaded.
