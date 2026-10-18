# B1/B2 frozen diagnostic: approved scope

No training, new seeds, U predictor, mask algorithm changes or threshold search.
Preserve all original checkpoint/selected-mask/prediction files read-only.

1. All8rates of B1/B2 seed66: each exact ACC-selected checkpoint/mask,
   one CHM scan/batch, replay original norm/head with full branch vs residual0.
   Capture CE,ACC,UA,W-F1, paired corrections/harms and residual/anchor norm.
2. Verify B2 rate0.5 BEST truly points to epoch13 in same full50epoch training
   trajectory (summary identity, config, last-state best references and hashes).
   Replay B2 epoch13 and epoch10 on B1 epoch13 rate0.1 exact savedmask.
   Reuse B1's first diagnostic prediction under that same mask. Compare same
   epoch plus within-B2 checkpoint-selection effect on the identical cohort.
3. B2 only: capture actual learnedS on valid strictpastW8pairs. Offline binary
   same-class classification and fixed10bin calibration. No calibration fit.
   Freeze intervene S=.5 =>D0,rho.5 for nonemptyhistory; keep E/R/CHM unchanged.
   One scan suffices; replay only branch fusion and norm/head for this control.

Correctness tests precede implementation. Require originalon savedlogits/class
parity, metadata/order/padding exact, inactiveGap/nohistory safety, no branch
state change, finite stats, all weights frozen, originalcheckpoint hashes intact.
Outputs go to independent osram_meld_related_history_diagnostic_20261018.

Interpretation boundary: branch-on/off measures the WHOLE residual, whose MLP
also consumes Local/Base/Gap, not a causal isolation of E or attentionR alone.
S=.5 is frozen sensitivity under a changed internal distribution, not a new
trained model's benchmark. Test labels used solely for offline scoring/pairs.
All results INTERNAL DIAGNOSTIC ONLY / Test-oracle seed66, no new performance claim.
