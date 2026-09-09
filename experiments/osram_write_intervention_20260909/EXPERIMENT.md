# Inference-only causal write intervention pilot

First task: IEMOCAP4 seed66, existing forward-only zero-slot checkpoint epoch33.
Evaluate rates0.0/0.1/0.3/0.5/0.7 with identical official test masks and weights.
Three modes reference, protected, global. No training, optimizer, new checkpoint
selection, hyperparameter search or changes to production model/trainer.

Reference returns exact existing block_write output. Protected applies
Delta - (Delta H) solve(H^T H + lambda_p I, H^T), lambda_p=0.001 fixed,
where columns are latest REAL observed keys for currently missing modalities.
Keys only update after the intervention; padding never updates history;
conversation/batch boundaries reset via backbone pre-hook. No future keys.

Global multiplies its own original Delta by ||protected Delta||/(||Delta||+1e-8)
per sample/head. This is norm-matched to a hypothetical protected update at
the SAME current state. After trajectories diverge, updates of the separately
executed protected/global models need not remain cross-trajectory norm-equal.
No eligible historical missing address is an EXACT identity for all modes,
including miss0, with no epsilon shrinkage. Positive ridge gives attenuation,
not exact nullspace projection or absolute protection.

Intervention is a temporary context-managed block_write override for eval only,
removed on exit. No trainable parameters, checkpoint keys, persisted buffers,
decay/read/JEPA/MMoE changes. Current read remains BEFORE write. This changes
inference dynamics of a frozen model and may cause distribution shift.

Report task W-F1 and retention separately. Improved old-association fit may
harm learning new observations; no automatic claim of task benefit. Pilot
results cannot establish five-seed robustness or a trained method's potential.
Checkpoint selection was historical eight-rate-mean Test-oracle: INTERNAL
DIAGNOSTIC ONLY, NOT A FORMAL PAPER RESULT.
