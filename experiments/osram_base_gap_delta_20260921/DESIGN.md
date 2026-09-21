# Base--Gap Delta Fusion

This experiment adds the independent `osram_readout_fusion="base-gap-delta"`
mode to the causal OSRAM backbone. It changes only the classification readout;
the memory scan, read-before-write order, key/value writes, Gap residual query,
JEPA, and predictor paths are unchanged.

For a valid utterance, the readout forms

```text
b       = W_b(Base)
g_m     = W_g(Gap_m)
delta_m = (g_m - b) * (1 - availability_m)
```

and feeds `[Local; b; delta_A; delta_T; delta_V]` to the existing Flat MLP.
Observed Gap slots are explicitly masked before projection and in the delta,
so arbitrary values in inactive slots cannot affect the output. Padding is
zeroed at the output.

The existing Flat adapter, local skip, and emotion normalization are reused;
the only new trainable parameters are the two context projections. The default
`flat` mode remains unchanged and old configurations without this value still
restore as `flat`.

This commit implements the mode and tests only. No large-scale training or
checkpoint reselection is included.
