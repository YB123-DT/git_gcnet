# Round 1: twenty completed readout-block experiments

INTERNAL DIAGNOSTIC ONLY: adaptive per-rate Test-oracle search, not validation-selected results.

All twenty candidates completed seed66 and100 epochs. Original Flat mean8 is81.068%; the strongest candidate, Perceiver IO, reaches80.818% (−0.250 percentage points). High-missing means are76.352% and76.126%, respectively. No candidate improves either aggregate; no candidate is promoted to seeds67/68 under the predeclared mean8 rule.

## Protocol and complete ranking

MOSI; original cfg84 no-JEPA objective and training protocol; cyclic random missing .0–.7; independent per-rate BEST checkpoints. Different rates may select different epochs, not one shared checkpoint. W-F1 is in percent; High averages .5/.6/.7. Deltas are percentage points. Ranking uses full precision. One seed cannot establish statistical significance or seed stability.

| Rank | Method | .0 | .1 | .2 | .3 | .4 | .5 | .6 | .7 | Mean8 | High | Δ Mean8 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| — | Original cfg84 Flat | 88.205 | 86.507 | 83.187 | 80.763 | 80.827 | 77.494 | 75.790 | 75.773 | 81.068 | 76.352 | 0.000 |
| 1 | perceiver_io | 87.799 | 85.615 | 83.091 | 80.982 | 80.679 | 76.753 | 76.488 | 75.137 | 80.818 | 76.126 | -0.250 |
| 2 | sheaf_hypergnn_diag | 87.481 | 85.754 | 82.838 | 80.563 | 80.861 | 77.130 | 75.415 | 76.163 | 80.776 | 76.236 | -0.292 |
| 3 | residual_gated_graph_evidence | 87.435 | 85.577 | 83.097 | 80.574 | 79.815 | 76.803 | 76.095 | 75.453 | 80.606 | 76.117 | -0.462 |
| 4 | egt_evidence | 88.227 | 86.016 | 82.365 | 80.435 | 79.416 | 77.041 | 75.023 | 76.136 | 80.583 | 76.067 | -0.486 |
| 5 | hamburger_nmf_full | 87.548 | 85.989 | 83.303 | 80.345 | 80.773 | 77.040 | 74.988 | 74.059 | 80.506 | 75.362 | -0.562 |
| 6 | pna_evidence | 86.822 | 85.606 | 82.088 | 80.538 | 80.842 | 75.967 | 75.843 | 75.948 | 80.457 | 75.919 | -0.611 |
| 7 | crate_mssa_ista_full | 87.137 | 85.725 | 83.257 | 80.514 | 79.824 | 76.287 | 74.975 | 75.721 | 80.430 | 75.661 | -0.638 |
| 8 | hyper_sagnn | 87.351 | 84.906 | 82.616 | 80.594 | 79.684 | 76.220 | 76.080 | 75.028 | 80.310 | 75.776 | -0.758 |
| 9 | capsule_dynamic_routing | 87.356 | 85.893 | 82.777 | 80.793 | 79.697 | 75.890 | 74.818 | 74.967 | 80.274 | 75.225 | -0.794 |
| 10 | equilibrium_aggregation | 87.058 | 85.409 | 82.815 | 79.983 | 79.944 | 75.933 | 75.505 | 75.036 | 80.210 | 75.491 | -0.858 |
| 11 | slot_attention | 87.161 | 85.553 | 82.204 | 80.635 | 79.858 | 76.378 | 74.708 | 74.658 | 80.144 | 75.248 | -0.924 |
| 12 | allset_transformer | 87.331 | 85.428 | 82.041 | 80.741 | 79.444 | 75.740 | 74.636 | 75.756 | 80.140 | 75.377 | -0.928 |
| 13 | capsule_variational_bayes | 86.673 | 85.280 | 83.145 | 79.807 | 79.372 | 76.733 | 74.893 | 74.818 | 80.090 | 75.481 | -0.978 |
| 14 | otke | 86.839 | 85.670 | 82.675 | 80.170 | 79.696 | 75.869 | 74.686 | 74.993 | 80.075 | 75.183 | -0.993 |
| 15 | graph_multiset_transformer | 86.579 | 84.932 | 82.325 | 80.386 | 79.800 | 75.967 | 75.280 | 75.322 | 80.074 | 75.523 | -0.994 |
| 16 | node | 87.418 | 85.169 | 82.037 | 80.591 | 79.294 | 75.556 | 74.760 | 75.048 | 79.984 | 75.121 | -1.084 |
| 17 | dgcnn_dynamic_edgeconv | 87.310 | 85.568 | 82.544 | 81.032 | 78.592 | 75.993 | 73.938 | 74.326 | 79.913 | 74.752 | -1.155 |
| 18 | ed_hnn | 86.624 | 85.507 | 82.003 | 80.080 | 78.753 | 75.580 | 73.943 | 74.393 | 79.610 | 74.639 | -1.458 |
| 19 | rrn_evidence | 85.794 | 85.202 | 81.810 | 79.247 | 78.697 | 75.847 | 75.384 | 74.685 | 79.583 | 75.306 | -1.485 |
| 20 | tabnet | 86.412 | 84.974 | 82.005 | 79.944 | 77.576 | 75.500 | 74.506 | 74.967 | 79.485 | 74.991 | -1.583 |

## Completion and provenance

Checked on biggpu on2026-10-04: run.completion_status returned complete for all20, auditing epochs1–100, eight BEST checkpoints, eight predictions, last training state, provenance and required artifact hashes. All40 local config/metrics files match remote SHA256. SUMMARY.json preserves full-precision scores and hashes; source files are in results/<method>/seed_66. Historical Flat was reused from BASELINE_AUDIT.json without retraining. Its original training commit was not recorded; later implementation commits must not be misrepresented as the training commit.

Coordinator4064601 retained stale EGT/CRATE admission-only failures despite complete artifacts. After exact command/start-tick verification, only this coordinator was terminated; independent control/queue.py was updated without changing the immutable training snapshot. Reconcile coordinator1980335 recorded all20 complete, entered awaiting_new_round and exited normally. No training was stopped or relaunched, and no checkpoints were deleted. Failed attempts remain in the historical record.

## Limits

These are exploratory single-seed results for adapted mechanisms, not claims that the original papers fail generally. Twenty-candidate Test-oracle screening is not a paper-ready evaluation. No winner met the declared promotion criterion, so this round does not expand seeds; later rounds are recorded separately.
