# Completed MELD Query/Decision diagnostic

INTERNAL DIAGNOSTIC ONLY. Seed66, eight fixed source checkpoints, 30epochs for
each of five small fits per rate. No CHM training or multi-seed expansion.

## Answer supported by this round

This restricted diagnostic did **not** establish that a new small readout or
current-conditioned Query correction extracts history information that beats
the equally sized Local control. It therefore does not justify identifying
Query as the main bottleneck or immediately changing Memory storage. Failure
of these fitted learners does not prove that Memory is uninformative.

Eight-rate means, same matched sample population, percent:

| Model | ACC | UA | W-F1 |
|---|---:|---:|---:|
| A: original frozen Flat | 60.674 | 35.436 | 56.805 |
| B_local: fitted Local control | 60.204 | 37.929 | 57.450 |
| B_donor: fitted wrong-history control | 59.064 | 36.367 | 56.179 |
| B_real: fitted true-history readout | 59.294 | 37.640 | 56.907 |
| C_donor: query-only fit, frozen B_donor | 59.620 | 37.349 | 56.927 |
| C_real: query-only fit, frozen B_real | 59.515 | 38.045 | 57.213 |

- B_real versus B_local: −0.910ACC, −0.289UA, −0.543W-F1 pp.
- B_real versus B_donor: +0.230ACC, +1.273UA, +0.728W-F1 pp. Real
  state has some advantage over the invalid-history control; this is not
  an advantage beyond Local in this learner family.
- C_real versus its exact B_real decoder: +0.220ACC, +0.405UA, +0.306W-F1 pp.
- C_donor versus B_donor: +0.556ACC, +0.982UA, +0.749W-F1 pp. Query
  adaptation can improve even wrong-history reads, so its gains alone cannot
  be attributed to recovering missing useful history.
- C_real versus B_local: −0.690ACC, +0.116UA, −0.237W-F1 pp. Compared with
  original A it increases UA/W-F1 but lowers ACC; Local control already
  produces most of this metric rebalancing without real Memory.

No single metric should hide these tradeoffs. CE also worsens slightly from
B_real1.319485 to C_real1.327631 at ACC-selected points. A stronger conclusion
would require genuine-history benefit beyond Local/Donor controls, rather than
merely changes in class balance or current-conditioned nonlinear capacity.

## Paired changes

Counts sum the eight rate-specific evaluations; these are20,879 sample-rate
exposures, not20,879distinct utterances. Labels, IDs and masks are aligned.

| Comparison | Corrections | Harms |
|---|---:|---:|
| A → B_real | 871 | 1159 |
| B_local → B_real | 886 | 1076 |
| B_donor → B_real | 1137 | 1089 |
| B_real → C_real | 179 | 133 |
| B_donor → C_donor | 357 | 241 |
| A → C_real | 888 | 1130 |

All rate/current-availability group details, corrections/harms, class recalls
and high-missing scores are in AGGREGATE.json and per-job runs/*.json. Active
patterns AT/AV/TV etc. here are **subsets of the random-mask evaluations**,
not separately trained or persistent-missing experiments.

## Population and reproducibility

- Full source A exactly reproduces seed66 ACC60.675287/UA35.436020/W-F1
  56.807056. This is not the accepted three-seed mean60.693.
- All9989trainutterances match a donor per rate. Test2610utterances match at
  seven rates; rate0.1has one unmatched row, so all compared groups use2609
  there. The tiny A full-versus-paired difference is this exclusion, not a
  model or mask change.
- Every source/decoder hash check passes. All owner gradients finite;
  zero-correction C reproduces B classes at each rate, maximum logit error
  1.90735e-6. Query role-specific gradient coverage is recorded, including
  no active Gap gradients at rate0.0.
- C_real selects epoch0 at rates0.0/0.3/0.6: **no benefit at these rates**.
  Epoch0 is deliberately included, so test-best C ACC cannot drop below B
  by selection construction. Even a positive difference is optimistic and
  not evidence of statistically significant or deployable improvement.
- Original source was cyclic-trained. Probe train views are one fixed
  offline GCNet view per rate; this constraint and different head capacities
  limit interpretation. The original inference structure stays unchanged.

## Cost and artifacts

biggpu GPU0/2, four independent rate jobs each. Each job123.02–129.00seconds;
total summed job time1004.95seconds, concurrent wall time about129seconds
after process initialization. Peak CUDA allocations2.3033GiB per job, not
host-card reserved usage. B152,781trainable parameters; C54,016.

Code snapshot16d0313; source original trainingb4a3138. LAUNCH.json contains
commands/PIDs/UUIDs. Remote run root:
`/data2/yb/remote_experiments/osram_meld_query_decision_20261014`.
Source checkpoints remain retained. New best/last small-head/query weights,
optimizer/RNG, masks/IDs/donor mappings and predictions stay at that root;
no weights, feature caches or secrets are uploaded to GitHub.

All40fits complete; no failed formal jobs.24CPUtests and real-batch GPU smoke
passed. No new D raw-history fit, extra loss, hyperparameter sweep, backbone
training or multi-seed extension was launched. Single-seed, test-selected
diagnostics do not support seed-level significance tests or paper claims.
