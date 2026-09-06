# OSRAM capacity variant: Out700

This is a single-variable capacity diagnostic relative to the H4 OSRAM+mean control. Only the OSRAM dimensions (heads=4, key_dim=32, value_dim=32, output_dim=700) changed. The OSRAM equations, predictor, JEPA loss, masks, cyclic schedule, optimizer, feature files, and evaluation protocol are unchanged.

This is an internal Test-oracle diagnostic, not a formal paper result. The variant is retained to determine whether a larger state/readout budget is useful before any architectural change.

Parameter counts are recorded in each seed's `metrics.json`; the H4 control is `experiments/osram_complete_20260906/`.
