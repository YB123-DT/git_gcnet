# Nested residual fixed-scale diagnostic

INTERNAL DIAGNOSTIC ONLY. No retraining. For each original MOSI Nested
checkpoint, scale exactly one of Local/Base/Gap-A/Gap-T/Gap-V correction by
`{0, 0.5, 1, 1.5, 2}` while holding the other four at one. Memory is scanned
once per batch and shared by all 21 unique readouts (the five scale=1 cases are
identical). Three seeds and the original eight test masks are replayed.
