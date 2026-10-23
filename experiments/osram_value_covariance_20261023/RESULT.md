# Frozen CHM Value covariance and task-response diagnostic

**INTERNAL DIAGNOSTIC ONLY. No training, optimizer step or stability loss.**

## Outcome and scope

The side ledger matches the conditional projected-Value sensitivity of original
joint CHM. Correlated read changes are measurable, but actual prediction effects
are very small at both pre-fixed amplitudes. These data do not establish a task
problem that the proposed stability loss would fix.

MOSI/MOSEI seed66 each use ONE original Flat miss0.3 checkpoint across evaluation
rates0.3/0.7, saved exact GCNet masks, and ALL test targets. **These are not
per-rate BEST scores or an eight-rate comparison.** Do not compare these rate0.7
scores with a separately selected rate0.7 checkpoint and call that a regression.

## Implementation and validation

```
A = I - eta Kbar P
Omega_new = A.T (alpha² Omega_previous) A + eta² P.T D² P
G = Q_effective.T (alpha² Omega_previous) Q_effective
kappa = diag(G)
```

Kbar/P are the actual embedded-beta joint write terms. Queries include actual
residualized Gap directions; inactive Gap/padding coordinates are zero. Sigma is
0.001/0.01 with16 draws, sharing unit Gaussian noise between amplitudes. Noise
enters projected historical Values BEFORE availability/sqrt-beta weighting.
Original production Block Write runs every noisy trajectory, preserving temporal
correlation. Local/Skip, K/Q, write coefficients and all parameters remain fixed.
Only forward512 reads change; backward512 remain zero; no-history noise is zero.

The complete same-time covariance is represented by per-head factors
`sigma² G_h ⊗ I_64`, with independent cross-head blocks under the noise assumption.
Flattening these blocks to original role-major coordinates requires permutation.
It is not empirical sentiment uncertainty or raw-input robustness. Different
times are NOT assumed independent in our actual Value-side replay.

| Dataset | Rate | Clean prediction max error | Actual-versus-linear max read error | MC covariance relative Frobenius error |
|---|---:|---:|---:|---:|
| MOSI | .3 | 0 | 8.31e-7 | 4.717% |
| MOSI | .7 | 0 | 8.59e-7 | 4.675% |
| MOSEI | .3 | 0 | 4.32e-6 | 4.580% |
| MOSEI | .7 | 0 | 3.67e-6 | 4.537% |

Linear realization error compares identical Gaussian perturbations through
production float32 writes with an independent double linear delta state.
MC covariance error instead compares finite16draw×64Value-coordinate estimates
with analytic G; it is not a4.7% algebraic error. The synthetic Jacobian-Gram
test agrees at1e-12 tolerance. Initial MOSEI execution_v1 failed clean parity
because of arithmetic shape rounding; it is retained but excluded. Preserving
production vector reads/full-sequence Flat resolved it without loosening the
1e-6 tolerance. See [NUMERICS.md](NUMERICS.md).

## Actual task effects, sigma0.01

Scores/counts average the SAME16 draws, not a best draw. Binary metrics exclude
y=0 and use pred>0; MSE includes neutrals. Correction/harm counts are PER DRAW.

| Dataset | Rate | N / binary N | Clean ACC | Clean W-F1 | Noisy W-F1 | Mean abs prediction change | Flip rate | Corrections / harms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSI | .3 | 686 / 656 | 80.7927 | 80.7412 | 80.7412 | 0.0004348 | 0% | 0 / 0 |
| MOSI | .7 | 686 / 656 | 67.2256 | 66.8981 | 66.8981 | 0.0006969 | 0% | 0 / 0 |
| MOSEI | .3 | 4659 / 3634 | 84.7001 | 84.5401 | 84.5566 | 0.00004137 | 0.01548% | 0.5625 / 0 |
| MOSEI | .7 | 4659 / 3634 | 78.1233 | 77.6210 | 77.6076 | 0.00005817 | 0.01204% | 0 / 0.4375 |

At sigma0.001 mean prediction changes are respectively0.00004348 /0.00006969 /
0.000004138 /0.000005816. All metrics/both amplitudes are in
[OVERALL.csv](completed_v2/OVERALL.csv). Sigma0.01 MSE changes are respectively
+1.219e-5 /-3.369e-5 /+1.264e-7 /+1.022e-6.

Projected-Value RMS is about0.59 in MOSI and1.23–1.26 in MOSEI. Sigma0.01 is
about1.69% and0.79–0.81% of those scales, not a large natural corruption test.

| Dataset / rate | Original wrong / correct | Correction rate given wrong | Harm rate given correct |
|---|---:|---:|---:|
| MOSI .3 | 126 / 530 | 0% | 0% |
| MOSI .7 | 215 / 441 | 0% | 0% |
| MOSEI .3 | 556 / 3078 | 0.10117% | 0% |
| MOSEI .7 | 795 / 2839 | 0% | 0.01541% |

Across16 draws, each MOSEI rate has only ONE distinct target whose polarity
ever flips: clean absolute predictions1.99e-6 at.3 and1.69e-5 at.7. These are
zero-boundary cases, not widespread systematic correction. MOSI has none.
Wrong targets show somewhat larger mean score changes, but correctness/margin/
availability strata are separate, not jointly controlled. Do not infer useful
or causal selectivity. All strata remain in the two AGGREGATE.json files.

## Read sensitivity is not task sensitivity

At sigma0.01, average read-noise RMS is0.0024–0.0031, versus mean prediction
changes0.00004–0.00070. These are different-dimensional spaces/scales, not a
dimensionless attenuation ratio. Independently computed first-order task
variance `grad_read(f).T (G⊗I) grad_read(f)` predicts actual mean squared score
change: actual/predicted ratios0.98661 /1.07631 /0.98850 /1.02475 for the four
rows above. Finite-draw variation remains; this is not exact equality.

Kappa alone does not rank task importance. MOSEI.3 head4 (one-based) has mean
active-role kappa0.12493 and unit task variance2.198e-5; head5 has higher
kappa0.18985 but variance8.705e-8. Head-only actual effects use just draw0,
sigma0.01; do not treat them as robust MC estimates.

Off-diagonal G is nonzero. For example MOSI.7 mean G(Base,Gap-T)=0.076091,
averaging all targets/heads INCLUDING inactive zeros. Full mean matrices with
that denominator are in [COVARIANCE_MEANS.json](completed_v2/COVARIANCE_MEANS.json).
Every target/head4×4 matrix and kappa remain in remote NPZ files.

## Provenance, cost and delivery

- Immutable inference source: `b8d4ccfef6e5824d37541fdcbf0833a0342322bf`.
- MOSI fixed miss.3 checkpoint epoch50 SHA256:
  `fd42eeaba7a7f24a748c6e3cf514f8b2cc8afe58cfc5b90968e0514fcb8aadc3`.
- MOSEI fixed miss.3 checkpoint epoch19 SHA256:
  `7905d518b4006d78f5f0d1139fc025ff190f9e1725da831d091292fd5575e0ab`.
- Both original checkpoints were Test-oracle selected. No new epoch,
  amplitude, threshold, configuration or draw was selected from this diagnostic.
- Server biggpu physicalGPU0, UUID`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`;
  GPU4 forbidden. Existing tasks were not signaled or modified.
- Original environment `/data2/yb/reproduction_workspace/envs/s0/bin/python`:
  Torch2.2.2+cu121, NumPy1.24.3, sklearn1.3.2.
- Full MOSI32.55sec/peak1788.7MiB; MOSEI192.49sec/peak2483.1MiB. Includes
  setup/identity checks on a shared GPU, not an isolated compute benchmark.
  Real batch smoke24.56sec/64.59sec, respectively.
- Remote root `/data2/yb/remote_experiments/osram_value_covariance_20261023`:
  immutable `source_b8d4ccf`; accepted `execution_v2/{mosi,mosei}`. Raw full
  ROWS/covariance/config/data manifests and failed-v1 artifacts remain remote.
- Small summary exporter SHA256:
  `9449c84b8aafb8e48385767cd3ac2c3c4fd83d53a00d2f8300fae0af564990c2`.
- [PROVENANCE.json](completed_v2/PROVENANCE.json) records checkpoint, mask,
  data/split/source/environment identities and hashes of full remote STATUS.

From the immutable source on biggpu, use a fresh output version:

```bash
bash experiments/osram_value_covariance_20261023/launch.sh \
  b8d4ccfef6e5824d37541fdcbf0833a0342322bf FRESH_VERSION
```

Verification:8 tests PASS locally and remotely; both real GPU smoke checks;
all target/covariance artifacts covered; saved miss.3 prediction/mask metadata
aligned; clean replay bitwise equal; frozen weight digest unchanged and no
parameter grads; independent sklearn W-F1 recomputation and unique target/no-
history checks PASS. Read-only math/shape review found no issues. No original
model file was modified. Changes are isolated diagnostic/tests/report files;
full-sequence/vector replay replaces the unnecessary alternative arithmetic.

Not tested:other seeds/rates, natural corruption, sentiment uncertainty, or
training the proposed loss. **The side ledger is validated, but excessive task
sensitivity is not established at these amplitudes.** Do not claim a successful
second Block, Memory irrelevance, or stability-loss efficacy. No regularization
run or larger-noise search was automatically launched.
