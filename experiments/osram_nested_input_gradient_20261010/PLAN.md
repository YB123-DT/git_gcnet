# Pre-Nested input gradients versus original Flat

INTERNAL DIAGNOSTIC ONLY

The user asks about gradient norms before the original Nested GNN. Measure the
task loss gradient with respect to the already read Local/Base/Gap tensors at
that interface. Flat is measured at its corresponding pre-readout interface.
These are activation gradients at existing trained checkpoints, not historic
training-time parameter gradients or forward feature magnitudes.

- MOSI original large Flat versus original `nested_gnn_rooted_evidence`.
- Existing seeds66/67/68, existing eight per-rate BEST checkpoints. No training,
  optimizer, checkpoint reselection, or weight changes.
- Each model uses its own actual Local and Memory trajectory. Identical raw
  conversations, masks, labels, sample IDs, original MSE and eval mode.
- Original causal scan runs once per model/batch. Then replay the exact original
  readout with detached Local/Memory leaves and autograd enabled. Local gradient
  includes both its original Skip and its adapter/Nested path.
- Memory gradient uses actual forward512 per slot, with backward512 constant0.
  Inactive Gap/padding are masked. Memory summaries exclude the no-history first
  utterance; Local summaries include all valid utterances. Record excluded counts.
- To avoid the batch-size divisor, compute gradients of summed valid per-row
  squared errors. Readout has no interaction between separate utterances, so each
  leaf gradient is its own utterance loss gradient. Report L2 and RMS (divide by
  sqrt dimension: Local256, each Memory slot512).
- Also record dy/dx Jacobian norms. MSE input gradients equal
  `2*(prediction-label)*dy/dx`; a larger gradient may reflect a larger prediction
  error rather than a more sensitive readout. Feature norms are separate columns.
- Report per-slot Local/Base/Gap-A/Gap-T/Gap-V and joint active Memory norms,
  per-rate/seed means, medians and paired Nested/Flat ratios. No causal claim that
  differences between independently trained checkpoints arise only from Nested.
- Verify full-forward versus replay predictions, saved W-F1, paired masks/sample
  IDs, loss/Jacobian scaling, mask/padding and frozen-state identity.
- Run on biggpu with an explicitly verified healthy GPU; physical GPU4 prohibited.
  Keep detailed per-utterance records remotely; small complete tables enter Git.
