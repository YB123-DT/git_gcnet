# IEMOCAP Flat Local/Base/Gap source combinations

Frozen inference on the retained cfg84 no-JEPA Flat checkpoints. Each
seed/fold/rate runs causal Memory once and replays all seven non-empty source
combinations at the original Flat readout. `L` includes Local Skip and the
adapter's Local slice, `B` is Base, and `G` is all three already-masked Gap
slots. IEMOCAP results pool the five held-out-session folds within each seed.

The diagnostic requires exact equality of the replayed `LBG` prediction,
labels, and availability against each source run's saved prediction artifact.
It is an internal per-rate Test-oracle diagnostic, not seven trained models.
