# OSRAM capacity variant: H8+V64+Output700

This is the only follow-up combination after the first capacity sweep. It increases heads, value width, and output width together while keeping key width at 32. No propagation, loss, mask, feature, optimizer, or predictor change was made.

Dimensions: `heads=8, key_dim=32, value_dim=64, output_dim=700`. The H4 OSRAM+mean control is `experiments/osram_complete_20260906/`; the other capacity controls are listed in `experiments/osram_capacity_sweep_20260906/`.

This is an internal Test-oracle diagnostic, not a formal paper result.
