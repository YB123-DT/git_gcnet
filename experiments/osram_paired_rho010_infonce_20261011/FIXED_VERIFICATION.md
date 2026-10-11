# Frozen persistent-missing evaluation: complete

Server biggpu GPU6 / GPU-e4cafb17-818e-216a-b94a-7440063a9153.
No training. Original rho.10 task-only C02 checkpoint files loaded strictly,
including training-only projector state. Its inference call count remains0.

Isolated remote root:
/data1/yb/remote_experiments/osram_paired_rho010_fixed_20261011.
Training backbone snapshot6e2a998, runner ec4fd9e. Four implementation hashes
matched original C02 provenance. No live InfoNCE training source was changed.
tmux socket paired_rho010_fixed_20261011, session evaluate, Python PID1742268.
Log evaluate.log; frozen records fixed_results.json. All six records complete,
686/656 total/nonneutral samples each; model/file hashes unchanged each time.

Same original fixed-pattern evaluator as earlier Flat/rho.50 records. Single
modality uses existing miss=.7 BEST; pairs miss=.3 BEST, no reselection based on
new pattern scores. Two local configuration/mapping tests passed. Report includes
all six results and negative outcomes; no new inference thresholds or seeds.
