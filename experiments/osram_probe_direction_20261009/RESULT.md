# Frozen Probe-direction decomposition of real history deletions

INTERNAL DIAGNOSTIC ONLY

Status: code and limited real-checkpoint verification passed; full diagnostic preparing. No training, new losses, new parameters or checkpoint selection.

Source: completed frozen Memory audit, original cfg84 LARGE Flat backbone seed66, eight inherited per-rate Test-oracle checkpoints, existing `preceding3_mean/B` probes. Probe66 primary,67/68 robustness, no best-seed selection. Replay ONLY saved `recent_vs_earlier_T` positions, both nearest and earlier arms; keep current Local/availability fixed. Do not generate new deletion plans.

Let R concatenate forward512 Base/Gap-A/Gap-T/Gap-V. Reconstruct D with its original training normalization and fixed random projection. At original R, take u=grad_R(D)/norm(grad_R(D)); project actual deletion delta onto u and its orthogonal complement in raw Euclidean coordinates. Feed original/full/projected/complementary R through the original frozen Flat and head. Inactive Gap and backward zero halves are untouched.

The rank-one gradient is a LOCAL PROBE-SENSITIVE direction, not a pure emotion direction. The complement may retain the same decodable information because D is nonlinear. Synthetic projected states need not lie on a real memory trajectory. F is nonlinear: projected and complementary prediction effects need not sum to the full effect. This is a sensitivity diagnostic, not a semantic mediation proof.

Verification:
- 19 unit tests passed: stored-plan identity, masks, finite differences/chain rule, exact probe restoration, projection algebra, original Flat replay, descriptive analysis.
- Real checkpoint .7,16 cases: frozen model/probe hashes unchanged; original F maximum replay error1.78814e-7; deleted F error2.23517e-7; original saved D prediction error2.38419e-7; reconstruction error0; orthogonality error1.65e-17. Current Local difference<=1.91e-6 (batch-shape numerical rounding). No tolerance relaxed.
- Check output: `/data2/yb/remote_experiments/probe-direction-check-ZqynZo/check_output`. Miniature-case scores are not performance findings.

Full command (biggpu healthy GPU2, UUID-pinned):

```sh
CUDA_VISIBLE_DEVICES=GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4 \
python -u -m experiments.osram_probe_direction_20261009.run \
  --output /data2/yb/remote_experiments/osram_probe_direction_20261009/results --gpu 2
```

The runner stores original/deleted full forward reads and each probe direction on biggpu, checks source hashes and numerical parity per rate, and automatically writes `results/summary`. Commit only aggregate reports, not raw tensors or weights. Report association before/after norm/lag/availability adjustment, four descriptive median-based groups, probe/task MSE, matched W-F1/ACC and corrections/harms, component sizes and nonlinear interaction. No claim yet.
