# Why close aggregate scores do not make V3 identical to Flat

INTERNAL TEST-ORACLE SCREEN; interim only,2026-10-21.

Existing remote history was read without new inference or training. At the
observation point MOSI had108 completed epochs; MOSEI11. Current runs continue
unchanged from source `f62310b`.

The actual computation is original Flat with corrected forward read inputs:

`LN(LocalSkip(L) + Flat([L, B + delta_B, masked(G + delta_G)]))`.

The shared Memory, original queries, Local Skip, original Flat and task head
remain. V3 adds individual same-source Base–Gap interactions and role-isolated
cross-source processing, not an identical copy of the Flat computation. Source
decomposition exposes finer Value-slot organization, but adds no new observed
features or new label supervision. Its useful incremental signal is unproven.

Existing training history provides evidence against an entirely inactive or
disconnected new branch:

- MOSI epoch108: all41 monitored parameter/input gradient paths nonzero and
  finite; last training-batch role correction mean norms(Base/Gap-A/T/V):
  `0.1847678572 / 0.1686799675 / 0.1572269499 / 0.1842161715`.
- MOSEI epoch6: all41 monitored paths nonzero; last training-batch role norms:
  `0.2610164583 / 0.4543109238 / 0.6600741148 / 0.3553708196`.
- MOSEI epoch11 last-batch corrections are all zero, with active_pair_mean=0.
  This is consistent with V3's required current-all-observed bypass; it does
  not establish an inactive branch throughout the epoch.

These diagnostics describe the last training batch, NOT epoch averages,
selected BEST weights or V3-on/off prediction changes. Without corresponding
original-read norms, absolute correction norms cannot be classified as small
or large relative to the inputs. Nonzero gradients do not prove useful
decision influence. Aggregate W-F1 proximity does not prove per-sample
predictions are unchanged.

Possible redundancy with Flat or joint co-adaptation remains a hypothesis,
not an identified cause. There is currently no evidence to claim residual
suppression, a Flat information ceiling, or a proven task-relevant role
interaction. Finish the already authorized two runs; no extra diagnostics,
ablation, seeds, parameter changes or new modules are launched by this note.
