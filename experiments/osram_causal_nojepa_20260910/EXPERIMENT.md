# Current causal OSRAM: remove JEPA

Approved task: remove all JEPA losses, not replace them with reconstruction.
Reuse existing `training_objective=emotion-only`: no missing predictor forward,
Teacher target forward, regression or InfoNCE supervision. Task loss only.
No model source edits; registered model architecture remains unchanged.

Control: inherited causal Flat eta=.6 MOSI seeds66–70,100epochs.
Treatment: same setup, only training_objective changes joint→emotion-only.
Train from scratch, cyclic rates0.0–0.7, same features/masks/LR/batch/optimizer.
No Cross-attention, B2, new reconstruction head, or extra loss.

Each seed × rate independently selects maximum Test W-F1 (earliest tie).
Reference scores extracted from full histories, not mean-selected checkpoint.
All averages descriptive only. INTERNAL DIAGNOSTIC ONLY, NOT FORMAL PAPER RESULT.

Reuse existing launcher with exclusive output creation and five workers on
GPU2(66,67,68),GPU3(69,70). No baseline retraining or separate smoke training.
Remote Python: /data2/yb/reproduction_envs/s0/bin/python3.10 on biggpu.
Remote root: /data2/yb/remote_experiments/osram_causal_nojepa_20260910.

Acceptance: five complete histories, all JEPA logged losses zero, matching test
mask hashes, per-rate selected epochs match independent extraction. Report all
rates, paired deltas, positive seeds, eight-rate and high-missing descriptive means.
