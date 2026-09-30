# Pre-launch verification

Remote smoke on biggpu GPU0, seed66/source miss0.7, one Stage2 epoch.
Original source epoch39. Original source checkpoint SHA256:
`1cc1c155c57c36879db60c6b63b668951ed679de6809e96e45bd2117d48dce68`.

- Epoch0 W-F1 exactly reproduces parent: 75.77317787056731%.
- Reference/current canonical label+availability rows match; subsequent ordered
  labels, availability and full mask hash remain unchanged.
- Only Gate parameters trainable: 231585.
- Original parameters AND buffers have unchanged SHA256:
  `d0ac1cd795c0fce8b620fe9e45e767a2a5ab733bad6c20fc9c2975c1666cde75`.
- After one epoch, Gate means moved from1 to Base0.962155, A0.964866,
  T0.962912, V0.966780. No near-bound saturation in this smoke.
- Final one-epoch W-F1 75.64437670597647%; best remains epoch0. This is
  implementation evidence, not a positive performance claim.
- best.pt and last.pt saved remotely; compact smoke metrics/history/provenance
  archived locally. Gate checkpoints require the referenced frozen source.

Test-fixture investigation: freezing a freshly initialized tiny Flat makes
Gate gradients zero because the original emotion_adapter final layer starts at
zero (osram.py), so the history branch cannot affect logits. This is not the
pretrained Stage2 condition. The fixture must first have a nonzero/trained
adapter before checking Gate updates; production source initialization is not
changed. The real-checkpoint smoke above independently verifies that Gate learns.

Smoke provenance records the source hash used for this run. Subsequent launcher
changes only strengthen failure admission control and progress provenance.
No full24-task accuracy result exists at launch time.

Final biggpu focused regression: `python -m pytest -q
tests/test_frozen_evidence_gate.py tests/test_local_evidence_gate.py`:
15 passed, one pre-existing PyG deprecation warning, 7.06 seconds.
Includes an actual three-step train_epoch with a learned-anchor fixture, Gate
updates, all nongate modules eval, and unchanged original state hash.
