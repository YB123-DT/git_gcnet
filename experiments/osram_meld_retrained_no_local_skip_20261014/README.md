# MELD retrained no-Local-shortcut ablation

This one-seed internal ablation retrains the causal no-JEPA Flat model from
scratch while disabling only the final `local_skip(Local)` term before
`emotion_norm`. The original Flat adapter still receives
`[Local, Base, Gap-A, Gap-T, Gap-V]`; OSRAM Memory and the missing-mask protocol
are unchanged.

Protocol: MELD seed66, 50 epochs, cyclic 0.0--0.7 training,
`gcnet-legacy-fixed-v1`, per-rate Test-oracle checkpoint selection.
