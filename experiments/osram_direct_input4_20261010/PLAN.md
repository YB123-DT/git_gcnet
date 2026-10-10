# Four approved non-residual evidence transforms followed by original Flat

INTERNAL DIAGNOSTIC ONLY. User approved CWN/GATv2/PNA/Perceiver direct-input runs.
No re-run of XCA/NPS or Nested. Each seed66 MOSI100epochs, original cfg84 Flat,
constantAdam1e-3, cyclic0.0–0.7, original task loss and per-rate BEST Test-oracle.
No Gate/auxiliary loss/completion/Memory changes. Preserve original Local Skip.

All: `[L,B,G] -> mechanism -> decoded[L',B',G'] -> original Flat Adapter`,
then `head(LN(skip(original L)+adapter([L',B',G'])))`. No external original-input
addition. Normal decoder initialization, not zero; first/empty-history rows retain
original input under existing adapter wrapper. Internal mechanism shortcuts remain.

GATv2/PNA: reuse five64d typed-role projections and exactly original one-layer
node-update math; expose outputs BEFORE old pooled readout. Decode Local64→256
and shared Memory64→512; discard unused old flatten readout in direct instances.
CWN: existing real33head-token complex/three updates unchanged; TokenAdapter
residual=False/zero_decoder=False. Same parameter count and RNG sequence as CWN.
Perceiver: existing real33head tokenizer128d, eight latent slots, three latent
processors unchanged. Decode each active token from shared latent set (Local query
uses original Local query projection; Memory query uses typed input token).
Normal local128→256 and eight shared-role per-head128→64 output decoders.
No persistent memory is added. This changes decoding query/readout interface.
Not parameter-matched removal-only ablations except CWN parameter count; do not
attribute results solely to residual removal. These are multiple module placements.

Files: priority40_relations.py (preserve old pooled path, expose node update),
meaningful_input_priority40.py (direct-role factory), priority40_registry.py /
meaningful_input.py (variant catalog without changing original40),
meaningful_input_new40.py / meaningful_new40_registry.py (CWN/Perceiver variants),
new direct_perceiver.py (reuse original encode/process/decode), core20/run.py
(approved experiment routing), tests/test_direct_input4.py, dispatch.py.

- [ ] Minimal contract tests RED for unregistered methods.
- [ ] Implement direct paths; preserve default operators and parameter init.
- [ ] CPU mask/first/padding/finitegradient/core-update and legacy graph parity.
- [ ] Remote CPU full-model test (existing torch_geometric environment).
- [ ] Scoped commit/push; sealed source/config/data/GPU/run manifests.
- [ ] Admit four independent runs on healthy GPUs, without disturbing others.
  Prefer GPU7 parallel; if insufficient reserve, queue in same server. GPU4 banned.
- [ ] Verify100epochs, artifacts and evaluation mask hashes; report all outcomes.
  No auto extra seeds/parameter sweeps.
