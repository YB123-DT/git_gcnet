# Dual-projector frozen completion diagnostic

This internal MOSI diagnostic keeps the cfg84 online projector/OSRAM/classifier
path trainable from its ordinary initialization. A separate frozen projector
bank and frozen `SourceOnlyM3Predictor` are loaded from the three-dataset JEPA
checkpoint. They encode only currently observed raw slots and predict missing
latents for `CompletedReadFusion`.

The complete raw missing modality is never used by the classification forward.
Predicted latents affect only the OSRAM read node; persistent OSRAM writes still
use real observed online latents and the original availability mask.

Checkpoint selection is the existing per-seed, per-rate Test-oracle internal
diagnostic protocol, with the same masks as the cfg84 no-JEPA control.
