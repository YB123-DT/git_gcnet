# C: paired task-only with rho=.25

Implementation/configuration prepared; formal training NOT started in this task.

The saved config copies the completed A (control) config exactly and adds only
`history_task_view2_weight: 0.25`. Loss becomes
`0.75 * task_view1 + 0.25 * task_view2`. Contrastive weight remains0.
Both task losses still include all valid utterances: no new anchor-only restriction.
View2 deletion probability.2, masks, RNG, two independent memory trajectories,
optimizer, epochs100, seed66, missing schedule, and evaluation remain unchanged.

CLI: `--paired-history-views --history-contrast-weight 0 --history-task-view2-weight 0.25`.
Global default rho remains.5; that branch retains the previous arithmetic order
exactly. Saved epoch diagnostics now record `task_view2_weight`; old checkpoint
configs without the field deserialize with.5. No new model parameters are added.

Verification: 13 paired-view/training/runner tests passed, including real CPU
forward/backward at rho.25, finite parameters, zero projector gradients for
task-only, invalid-rho rejection, and existing RNG/mask/memory-isolation tests.
Config comparison verifies no changes beyond the single rho field. Future runs
belong on biggpu GPU6 with a fresh output directory; never overwrite A/B runs.
