# Nested Gap residual Gate implementation plan

**Approved design:** keep the original Nested GNN and all original readout paths.
Only change `Gap_m + delta_Gap_m` to `Gap_m + g_m * delta_Gap_m` for A/T/V.
One scalar per utterance/modality, shared across all eight actual memory heads.
`g=2*sigmoid(z)`, zero last gate layer gives g=1. No new loss or training stage.

**Scope:** implement and verify; no formal training authorized by this plan.
Keep original methods/checkpoints valid. New method:
`nested_gnn_gap_residual_gate`. Technology: existing PyTorch, no dependency.

Files: new `gcnet_missing_m3/nested_gap_gate.py`; small dispatch additions in
`meaningful_input_new40.py` and `meaningful_new40_registry.py`; optional diagnostics
forwarding in `meaningful_input.py`; new `tests/test_nested_gap_gate.py`.

- [x] Write tests before code. Check registry, identical original/RNG initialization,
  identity gate with nonzero trained decoders, residual-only .5/.75/1.5 gates,
  public mask/first-turn/backward-half behavior and finite gate updates.
- [x] Run `/home/yangbin/miniconda3/envs/msa_extract/bin/python -m pytest -q
  tests/test_nested_gap_gate.py`; confirm missing-method failures.
- [x] Implement shared projections (Local/evidence/delta to64), type embedding16,
  `[q;k;d;type;availability]` LayerNorm → Linear32 → GELU → Linear1.
  Gate inputs and outputs use safe active masking. Isolate gate initialization RNG.
  Insert only before decoded Gap residual addition, preserve Base/Local.
- [x] Rerun new and original Nested contract tests. Record parameter count, gate
  active means, active counts, residual norm and gated residual norm.
- [ ] Save implementation report, scoped Lore commit and push current github branch.

No historical labels, second query, completion, persistent mix, test-derived rules,
or new loss. A lower gate is not assumed to improve W-F1. Old checkpoint loading
into a new model must explicitly account for added gate parameters, not silently
ignore arbitrary missing keys.
