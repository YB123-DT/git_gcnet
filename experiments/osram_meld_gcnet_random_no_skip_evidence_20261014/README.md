# MELD GCNet-random no-skip evidence ablation

Frozen inference on a seed-66/67/68 no-Local-skip model trained with
`gcnet-random-mask-v1`. Each seed is evaluated in an independent output
directory using its own checkpoints and saved availability artifacts.

For each missing rate, the evaluator loads the corresponding per-rate
Test-best checkpoint. The exact availability realization saved by that run's
final prediction artifact is replayed for all four conditions:

- `LBG`: Adapter Local + Base + Gap;
- `LG`: Adapter Local + Gap (Base zeroed);
- `LB`: Adapter Local + Base (all Gap slots zeroed);
- `L`: Adapter Local only (Base and Gap zeroed).

Memory is scanned once per batch. The interventions occur only in the captured
Flat Adapter input. Full replay must reproduce the saved predictions exactly.
Because the original sampler is stochastic and the availability that first set
the Test-best value was not stored, absolute scores correspond to the saved
paired replay mask; contribution claims use paired differences.
