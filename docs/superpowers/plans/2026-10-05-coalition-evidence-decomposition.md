# Coalition Evidence Decomposition (CED) implementation plan

Goal: implement the user's observed-only coalition Möbius block and launch one
MOSI seed66 screening run on biggpu GPU6; no additional loss or sweep.

Architecture: share current ModalityProjectors and mean ObservedSetEncoder
fusion across all nonempty subsets of the actual observed set. Define F(empty)=0.
I(S)=sum(T subset S)(-1)^(|S|-|T|)F(T). Preserve three order slots:
c1=mean P1(I(single)), c2=mean P2(I(pair)), c3=P3(I(ATV)), active only.
P1/P2/P3: LayerNorm256 -> Linear256 -> GELU. Pout:
LayerNorm771 -> Linear771,256 -> GELU, normal initialization.
Output is directly L_CED, not an anchor-plus-residual and not sum of dividends.
Missing orders/padding are zero after bias-containing operations.
Availability is included in Pout. Latents supplied to Value path remain the
original observed-only modality latents; node-dependent keys/queries change
because the current encoder output changes. Memory formulas and Flat stay intact.

Files/ownership:
- `gcnet_missing_m3/coalition_mobius.py`: isolated shared-coalition encoding,
  transform, order slots, pooling/output, diagnostics and opt-in attachment.
- `tests/test_coalition_mobius.py`: exact transform, mask safety, shared dropout,
  observed-only projection, independent order slots, finite gradients.
- Main integration: `model.py`, `train_gcnet.py`, registry validation;
  `tests/test_ced_integration.py`; dedicated runner and RESULT.md.

- [ ] Write focused tests first; demonstrate missing new block fails.
- [ ] Implement math/block; full coalition and proper subsets share projector
      realization and fusion dropout mask for each current utterance.
- [ ] Add default-off `--osram-ced-block`; attach after original initialization
      inside fork_rng. Reject combinations with other optional methods.
- [ ] Verify `sum(I(S))=F(O)` only as a diagnostic, not as final readout;
      additive toy set function has zero higher orders; AV uses only A/V/AV.
- [ ] Run focused CPU checks in the existing biggpu environment plus two real
      small training updates; disabled outputs/weights/RNG remain unchanged.
- [ ] Commit/push; seal immutable source; keep exact seed66 baseline optimizer,
      100 epochs, cyclic .0–.7, original masks and per-rate BEST Test-oracle.
- [ ] Verify actual optimizer step, PID/GPU/log/config and record launch.

Commands: CPU `python -m pytest -q tests/test_coalition_mobius.py
tests/test_ced_integration.py`; remote run `python -m
experiments.osram_ced_20261005.run` with original reference/data manifest and
an independent output directory. No new dependency, completion, JEPA, gate,
dual-view or extra loss. INTERNAL DIAGNOSTIC ONLY.
