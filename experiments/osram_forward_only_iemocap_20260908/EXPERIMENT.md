# IEMOCAP forward-only OSRAM diagnostic

Only change: osram_bidirectional=False; reverse slots zero; no reverse scan.
Inherit per-seed configs from osram_iemocap4_20260906 and
osram_iemocap6_20260906. No baseline reruns, no model/loss changes.
Both tasks: Session5, seeds66–70, cyclic eight rates 0.0–0.7,
100 epochs, batch32, LR1e-3, H8/key32/value32/output700.
Report per-rate Test-oracle maxima, not formal paper results. Official IEMOCAP
uses the same held-out session for validation and test.

GPU5: seeds66/67/68; GPU6: seeds69/70. Each lane runs 4-class then 6-class.
Ten training tasks total, five concurrent. Do not use GPU4 or disrupt GPUs0–3.
Feature root: /data2/yb/paper/GCNet_TPAMI_modality_jepa_20260818/dataset/IEMOCAP/features
Remote output: /data2/yb/remote_experiments/osram_forward_only_iemocap_20260908

Inherited implementation verification: 192 related tests passed for the same
single-direction switch, including future perturbation invariance. This turn
adds only an inherited-config launch script, not another backbone modification.
