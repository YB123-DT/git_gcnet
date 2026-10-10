# Verified non-residual input transform followed by preserved Flat

INTERNAL DIAGNOSTIC ONLY. Repository search; no new training/inference.

Two completed experiments directly match the requested placement:

| Method switch | Seed66 mean8 W-F1 % | High % | Decoder initialization |
|---|---:|---:|---|
| nested_gnn_direct_evidence |77.315007|71.311849|zero|
| nested_gnn_direct_random_evidence |80.176918|74.800208|ordinary nn.Linear|

Both are one-stage MOSI100 epochs, original cyclic missing and per-rate BEST
Test-oracle. Scores are not three-seed averages. Original residual Nested
reference:80.992147/76.077229; original Flat:81.068095/76.352251.

Actual valid-history computation:

`[L,B,G] -> Nested graph -> decoded [L',B',G'] -> original emotion_adapter`.
Final output remains `head(emotion_norm(local_skip(original L) + adapter([L',B',G'])))`.

There is no `L+decoded_L` or `C+decoded_C` input residual in these variants.
The original Local Skip addition is deliberately retained. This is therefore
NOT a network without any addition/internal shortcut, NOT deletion of Flat,
and NOT a transform of the OSRAM persistent state. Empty-history/first rows
retain raw inputs via MeaningfulInputAdapter; inactive Gap slots remain masked.

Implementation evidence:
- gcnet_missing_m3/meaningful_input_new40.py: TokenAdapter.forward branches on
  `self.residual`; false returns only decoded local/evidence.
- gcnet_missing_m3/meaningful_new40_structure.py: builds these two variants
  with residual=False; random variant also sets zero_decoder=False.
- gcnet_missing_m3/meaningful_input.py: preserves first/empty-history rows.
- gcnet_missing_m3/osram.py: meaningful_input_mode passes changed evidence to
  emotion_adapter while local_skip still receives original local.

Completed result sources:
- ../osram_nested_direct_20261008/RESULT.md, SUMMARY.json.
- ../osram_nested_direct_random_20261008/RESULT.md, COMPLETED_RESULT.json.
- ../osram_readout_path_audit_20261009/RESULT.md.

This is a verified exact-match subset, not a claim that no other input transforms
exist. Naming something an input replacement is insufficient: many input cores
still internally add decoded corrections to their original evidence.
