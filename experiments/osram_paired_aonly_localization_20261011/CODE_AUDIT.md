# Historical implementation audit

Read-only comparison of rho.50 control and rho.10 C02. No identified mask/loss
implementation bug that explains A-only polarity bias.

- Historical6e2a998 train_gcnet.py _paired_history_loss uses .9*task1+.1*task2.
  Recomputing100epochs from saved logs: maximum discrepancy4.64e-6 for rho.10,
  3.82e-6 for rho.50, consistent with float rounding.
- Both View1 and View2 mask SHA256 match at every one of100epochs.
- model.py/osram.py/paired_views.py SHA256 match. train_gcnet change introduces
  rho field, validation, weighted task formula, diagnostics and CLI.
- Audio/Text/Visual order and repeated feature-dimension masks agree between
  training and fixed evaluation. View2 only removes observations from incomplete
  View1. Neither training view exposes previously missing features.
- Each OSRAM _scan allocates zero initial Memory. Frozen evaluation runs one
  view and calls no contrastive projector. Task-only contrastive weight is0.
- Adam.001, batch32, dropout, clip norm1, MSE/sample-mean,100epochs,200 optimizer
  updates and View2 drop probability.2 agree. rho changes gradient composition;
  clipping-before/after norms were not retained, so clipping cannot be assigned
  as the causal explanation.

Sources: original paired_history_views/results/control and paired_history_rho025/
results/C02 config/history/PROVENANCE files. Note epoch-mean drop fractions differ
from total removed-bits/total observed-bits; the latter is16.1745% for both runs.

Interpretation remains bounded: changing rho changed the optimization path and
the selected checkpoint, while keeping training observations identical. This
audit does not establish exactly which updates produced the failure.
