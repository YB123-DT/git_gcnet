# Frozen Probe-direction decomposition of real history deletions

INTERNAL DIAGNOSTIC ONLY

Status: COMPLETED on biggpuGPU2,2026-10-09 13:16:25–13:17:55UTC; final summary verified after process exit. Source snapshot `d825316`, persistent tmux `probe_direction_20261009`, PID1811618. All8rates and three existing probes completed;25,734 rows =4,289 eligible target/rate pairs × two deletion arms × three probe seeds. No training, new losses, new parameters or checkpoint selection. All48 recorded artifact hashes independently rechecked.

## Main result

The decoded historical-mean score and original task prediction co-vary under actual Text deletion, but the local probe-gradient component alone produces almost no original-task response. The orthogonal component preserves nearly all of the original task response. This limits the claim that this PARTICULAR local gradient direction explains the task head's sensitivity; it does not prove the model ignores historical sentiment.

Primary probe66, nearest-eligible Text deletion, equal mean over8rates. Classification metrics are ONLY the matched eligible targets (4,289 target/rate pairs,4,133 nonneutral), NOT the full-test-set eight-rate benchmark:

| Readout condition | Mean absolute change of D | Mean absolute change of original F | W-F1 (%) | F MSE change | Corrections / harms |
|---|---:|---:|---:|---:|---:|
| Original R |0|0|81.991967|0|0 / 0|
| Full actual deletion R+delta |0.411532|0.160370|80.974292|+0.036188|120 / 161|
| Probe-direction component only |0.404098|0.000239|81.991967|-0.000040|0 / 0|
| Orthogonal component only |0.026993|0.160168|80.952926|+0.036087|119 / 161|

These absolute-change means are not additive attribution fractions. The full/projection/complement mean raw displacement norms are7.565072 /0.228843 /7.560859. In particular, the projection has much smaller raw magnitude; this is not a norm-matched comparison. Nonetheless it changes the decoder substantially while leaving F nearly unchanged. Changing D along its gradient is partly built into this construction; the empirical observation of interest is F's weak response, not D's expected sensitivity.

For probe seeds67/68, projected-component mean absolute F changes are0.000260 /0.000243 and projected F MSE changes -0.000043 /-0.000048, respectively. All three probes produce zero projected-component polarity corrections/harms in the nearest-deletion arm. This is probe-initialization robustness on ONE frozen backbone, not independent model-seed replication.

For the earlier-Text deletion control (primary66), full/projected/complement F MSE changes are -0.001813 /-0.000010 /-0.001818. Mean absolute F changes are0.051428 /0.000082 /0.051362. The same directional separation appears, without assuming that every historical deletion must worsen predictions.

## Association and controls

Primary66 nearest deletion: signed DeltaD/DeltaF Pearson correlations range0.753–0.835 across rates and remain0.753–0.836 after adjustment for log1p(raw displacement norm), lag and availability. Absolute-change correlations range0.503–0.716 but fall to -0.016–0.312 after adjustment. Thus raw co-variation alone is not evidence of using the same representation direction.

Four descriptive groups use within-rate/seed/arm medians, NOT fitted thresholds: for primary66 nearest deletion, equal-rate fractions are37.53% small/small,12.53% small-D/large-F,12.53% large-D/small-F,37.42% large/large. Thresholds and all cells are retained in `summary/quadrants.csv`. These arbitrary descriptive median groups are not success criteria.

Detailed results: [summary](summary/RESULT.md), [all statistics](summary/SUMMARY.json), [associations](summary/associations.csv), [per-rate and availability](summary/per_rate.csv), [conversation-bootstrap intervals](summary/cluster_bootstrap.csv). No aggregate significance claim is inferred from the displayed means. Nonadditive F response mean absolute magnitude is7.63e-5 for primary66 nearest deletion and is reported rather than forcing additive contributions.

Source: completed frozen Memory audit, original cfg84 LARGE Flat backbone seed66, eight inherited per-rate Test-oracle checkpoints, existing `preceding3_mean/B` probes. Probe66 primary,67/68 robustness, no best-seed selection. Replay ONLY saved `recent_vs_earlier_T` positions, both nearest and earlier arms; keep current Local/availability fixed. Do not generate new deletion plans.

Let R concatenate forward512 Base/Gap-A/Gap-T/Gap-V. Reconstruct D with its original training normalization and fixed random projection. At original R, take u=grad_R(D)/norm(grad_R(D)); project actual deletion delta onto u and its orthogonal complement in raw Euclidean coordinates. Feed original/full/projected/complementary R through the original frozen Flat and head. Inactive Gap and backward zero halves are untouched.

The rank-one gradient is a LOCAL PROBE-SENSITIVE direction, not a pure emotion direction. The complement may retain the same decodable information because D is nonlinear. Synthetic projected states need not lie on a real memory trajectory. F is nonlinear: projected and complementary prediction effects need not sum to the full effect. This is a sensitivity diagnostic, not a semantic mediation proof.

Verification:
- 19 unit tests passed: stored-plan identity, masks, finite differences/chain rule, exact probe restoration, projection algebra, original Flat replay, descriptive analysis.
- Real checkpoint .7,16 cases: frozen model/probe hashes unchanged; original F maximum replay error1.78814e-7; deleted F error2.23517e-7; original saved D prediction error2.38419e-7; reconstruction error0; orthogonality error1.65e-17. Current Local difference<=1.91e-6 (batch-shape numerical rounding). No tolerance relaxed.
- Check output: `/data2/yb/remote_experiments/probe-direction-check-ZqynZo/check_output`. Miniature-case scores are not performance findings.
- Full-run maxima: original F replay error7.15256e-7; deleted F8.34465e-7; saved D prediction1.07288e-6; reconstruction0; orthogonality5.55112e-16. Main model and all probe hashes unchanged. No negligible-gradient cases. Source and inactive-slot checks passed; no new deletion plan or parameter update was performed.

Full command (biggpu healthy GPU2, UUID-pinned):

```sh
CUDA_VISIBLE_DEVICES=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
python -u -m experiments.osram_probe_direction_20261009.run \
  --output /data2/yb/remote_experiments/osram_probe_direction_20261009/results --gpu 2
```

The runner stored original/deleted full forward reads and each probe direction on biggpu, checked source hashes and numerical parity per rate, and automatically wrote `results/summary`. Only aggregate reports and provenance are committed, not raw tensors or weights. The measurement supports decoupled local sensitivities under this construction, not pure-semantic mediation or the absence of history use by the original model.
