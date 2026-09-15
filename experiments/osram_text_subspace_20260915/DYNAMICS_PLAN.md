# Frozen subspace magnitude/gradient audit

No optimizer steps, no training, no loss/temperature changes. Reuse seed66 two-update Stage1 checkpoint. Reconstruct R0 through the exact original seed/data/projector initialization sequence; verify its first-batch R sentiment/prediction and Q prediction gradients against archived smoke. Do not call this a saved R0 snapshot.

1. Compare centered covariance effective rank for Teacher Text256, initial random R0→32, selected R→32 on identical train and validation rows. Record std and covariance too.
2. On the same original first training batch, collect separate R gradients for sentiment/predictability/variance/covariance, six pairwise cosines, total-R gradient and global R/C/Q clip coefficient (diagnostic only). Repeat at selected R/C/Q without replaying optimizer steps.
3. Reconstruct original fresh Stage2 Student and exact dropout/batch realization, assert emotion/reg/NCE reproduce archived smoke. Measure emotion vs .1JEPA on shared OSRAM and ObservedSetEncoder parameter coordinates, both separately and combined. Exclude private classifier/MMoE/Teacher/R; report common used parameter support and zero-gradient cosines as null, not zero.
4. Save JSON and report, verify hashes/grad fields/RNG safety, tests, diff check, commit/push. Stop. These are single-batch diagnostics, not generalization or causal root-cause proof. Stage1's local Teacher A/V predictability is not Stage2's full causal information set.
