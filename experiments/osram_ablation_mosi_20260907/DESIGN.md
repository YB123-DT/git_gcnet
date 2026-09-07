# OSRAM mechanism ablation design

This is an internal CMU-MOSI diagnostic.  It isolates which OSRAM readout
path reaches the emotion head while keeping the frozen features, cyclic
mixed-rate schedule, Student/EMA/JEPA stack, optimizer, seeds, and test-oracle
protocol fixed.

The inherited Full OSRAM condition is the H8/32/32/700 run in
`experiments/osram_heads8_out700_20260906/`; it is not retrained.

## Conditions

| Condition | Local utterance path | Base context | Missing-specific gap context |
|---|---:|---:|---:|
| Full OSRAM (inherited) | on | on | on |
| Local-only | on | off | off |
| Local+Base | on | on | off |

The ablation switch masks only the corresponding OSRAM context slots before
the emotion adapter and structured missing-latent predictor.  The associative
scan and parameterization are unchanged, so this is a mechanism readout
ablation rather than a parameter-matched model.

## Interpretation

Local-only tests whether the current utterance representation is sufficient.
Local+Base tests whether ordinary bidirectional memory helps independently of
missing-pattern-specific reads.  The difference between Full and Local+Base
is the incremental contribution of the A/T/V gap slots.

The result uses Test-oracle checkpoint selection for diagnosis only and must
not be reported as a formal benchmark result.
