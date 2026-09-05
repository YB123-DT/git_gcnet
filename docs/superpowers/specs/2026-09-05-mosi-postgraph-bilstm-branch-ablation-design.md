# MOSI Branch-specific Post-graph BiLSTM Ablation Design

## Question

Does branch-specific post-graph recurrent smoothing harm MOSI after Temporal or
Speaker graph propagation?

## Variants

- Control: both branch Post-BiLSTMs enabled; inherit commit `62208ae` results.
- T-off: bypass only `graph_net_temporal.grufusion`.
- S-off: bypass only `graph_net_speaker.grufusion`.

The two treatments are separate. `RGCNConv`, second `GraphConv`, graph
topology, branch addition, Text-anchor fusion, JEPA, and emotion readout remain
unchanged.

## Exact bypass

`GraphNetwork` normally maps a `D_h` conversation sequence through a
bidirectional LSTM to `2*D_h`, then through the existing `linear: 2*D_h -> D_h`.
In a bypassed branch, concatenate the unchanged sequence with itself to form
`2*D_h`, then use the same `linear`. This preserves the readout, dimensionality,
parameters, state keys, optimizer construction, and RNG behavior while
isolating recurrent mixing. The unused BiLSTM parameters receive no gradients.

Add `postgraph_bilstm_ablation` with choices `none`, `temporal`, and `speaker`.
Default `none` exactly preserves current behavior. Selective bypass is allowed
only with `postgraph_sequence_mode=independent`.

## Experiment

Run CMU-MOSI seeds 66--70 for both treatments, 100 epochs, cyclic mixed rates,
Text-anchor fusion, GraphConv2 enabled, and independent per-rate Test-best epoch
selection. Report each treatment against the inherited control and against the
other treatment. Label all results Test-oracle diagnostics.
