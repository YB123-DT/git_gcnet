# Locked top-three seed confirmation

INTERNAL DIAGNOSTIC ONLY

Selected before confirmation from completed C01–C20 methods, ranking seed66
8-rate BEST Test-oracle W-F1; controls excluded. Unfinished methods excluded.

| Method | seed66 8-rate (%) | high (.5/.6/.7) (%) |
|---|---:|---:|
| C20 CAGrad three-exit supervision |80.645543|75.883865|
| C07 Recurrent Independent Mechanisms |80.513719|76.256848|
| C08 TTT-MLP |80.089774|75.200594|
| Original Flat |81.068095|76.352251|

Reuse completed seed66; six authorized new runs = each method × seeds67/68.
From scratch, same-seed original Flat reference config. Those reference configs
were checked to differ only in seed. 100 epochs, all method-specific coefficients,
task objectives, network dimensions, Adam/lr/batch and missing/evaluation protocol
remain fixed. No new hyperparameter sweep or GPU smoke. No new control training.

Deployment must copy original immutable source_00522bd; only run.py and
multiseed.py differ. Record original_model_commit and wrapper commit in SNAPSHOT.
GPU6 only. C20/C07 admitted gradually using real peaks +20%+512MiB and 2GiB
headroom (3GiB initial reservation). C08 runs singly after the original batch
queue and the confirmation light runs drain, with 18GiB admission reservation.
No original process is stopped; failed jobs are recorded, not blindly relaunched.

Status: implementation complete; deployment pending. Final analysis should show
all three seed-matched deltas vs Flat, mean/sample SD, eight per-rate means and
high-missing means. Do not replace missing runs with best single-seed scores.
No significance or formal validation-selected performance claim.
