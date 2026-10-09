# Flat256 + Nested704 implementation plan

Approved user request: implement and run one MOSI seed66/100epoch comparison. Execute inline in the current dedicated worktree, preserving unrelated edits.

Execution record: all five implementation/launch steps below completed. Red test rejected missing method; CPU and actual-batch CUDA checks passed; code a3e4a9b sealed and launched onGPU3; first formal epoch and effective config verified. Only completion of100epoch training and final artifact analysis remain pending. Checklist below retains the original plan text.

- [ ] Add `check.py`: first require `candidate_config(reference,'small_flat_nested_dim704')`, assert only adapter256 and Nested name change; then assert full model13560676 and module8051715 parameters, zero-init equality, padding/inactive Gap/first-turn behavior, finite updates. Run before registration and observe unsupported method.
- [ ] Register `nested_dim704: {'dim':704}` in `meaningful_new40_registry.py`; add `small_flat_nested_dim704: nested_dim704` to existing small-Flat runner mapping. Forward existing adapter width through the reusable exact-model builder so the count check uses the actual requested width. No new graph math.
- [ ] CPU check on existing remote temporary source; then one actual maximum-valid-utterance train batch using original task/mask preparation at epoch7 and paired evaluation on healthyGPU3. Record peak CUDA memory; do not alter batch/loss/protocol.
- [ ] Commit/push scoped changes. Extract immutable git archive, seal source, invoke existing single-job dispatcher onGPU3; confirm live PID, effective config, first completed epoch and preserved checkpoint outputs.
- [ ] Update RESULT with verification and launch identity, commit/push. Training results remain pending until100epochs and artifact verification; no extra seeds/configs authorized.

Command template: `python -m experiments.osram_nps_local_20261009.dispatch --method small_flat_nested_dim704 --root /data2/yb/remote_experiments/osram_nested_equal_budget_20261009 --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json --gpu-index 3 --gpu-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`.
