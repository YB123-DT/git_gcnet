# TASR: finite implementation and screening contract

INTERNAL DIAGNOSTIC ONLY. Parent source: ea27c084ad21d38a01025ed0f8fbaccb6b1dcb49,
feature/ecerc-features. Original model/checkpoint protocols remain protected.

## Algebra and unchanged inference paths

Let P=(Kbar^T Kbar+ridge I)^(-1)Kbar^T, using the ORIGINAL joint weighted
columns Kbar/Vbar (availability times sqrt(beta)). The authoritative update
remains M'=D+eta(Vbar-D Kbar)P, D=alpha M. Tracking alone uses
H'_s=alpha H_s+eta(Vbar[:,s]P[s,:]-(alpha H_s Kbar)P), all with the SAME solve.
Summing over s reconstructs the original update by linearity, up to floating
point roundoff. No sequential writes, new state substitution, or new queries.

The original residualized Gap query reads each decayed source BEFORE writing.
These three reads reconstruct the original forward Gap. Sources mean additive
Value-slot contributions to a coupled update, NOT pure-modality memories or
semantic causal attribution. Future other-source writes may erase old sources.
Backward-half slots remain zero. Padding neither decays states nor advances
the trajectory clock. No histories or clocks are shared across conversations.

## Adapter and controls

Count, coverage, age, ever-seen are strictly-past, per conversation/source.
Never seen: age=0 and ever=0; most recent previous utterance: age=1.
Local256/read512 project separately to64; source/target embeddings8 each.
The151-d concatenation (64+64+3+4+8+8) passes LN/Linear64/GELU/Linear1.
Final Linear zero-init; gamma=.25 tanh(score). Only missing-target forward
Gap is corrected by sum_s gamma_ms G_ms. Original Base, Local/Skip, query,
joint write, Flat1600, task head, and CE stay intact. Initialization uses
fork_rng so original/downstream parameter initialization and RNG are unchanged.

F0: fixed original Flat, no new training. F1/F2 each add59,423 parameters.
F1 presents original Gap/3 to EVERY source slot, without dropping a synthetic
slot because that actual source was never seen; otherwise this would not be
the specified neutral control. F2 presents actual source reads and masks
never-seen zero sources. Both share all trainable shapes and initialization,
Local, trajectory metadata, zero output initialization and training budget.

Coefficient bounds do not imply ||delta Gap|| <= .25||Gap|| when sources
cancel. Actual correction/read norm ratios must be reported, not assumed.

## Predefined finite screening

MELD seed66 and IEMOCAP6 seed66/fold5 ONLY. Each dataset has ONE protected
original miss=.3 anchor for ALL random rates/persistent patterns and controls.
These anchors were originally selected by Test-oracle: no independent-test
generalization claims. Do not compare the new fixed-anchor mean to an old
per-rate selected-checkpoint mean as if they were paired results.

Freeze every original parameter and keep original model in eval mode. Train
only Adapter (Adam3e-4, wd1e-5,30epochs,batch32). Every step runs independent
zero-initialized trajectories on random View1 (original GCNet v1 host/guest
sampler, unchanged) and Persistent View2 from COMPLETE raw training features.
One uniform six-pattern draw per conversation; no nested deletion, no InfoNCE.
Loss=.8CE_random+.2CE_persistent, both on all valid utterances. Fixed seed66;
dedicated persistent RNG must not perturb original random-mask RNG.

MELD: official development conversations select a single best adapter by
macro8-rate validation ACC; lower meanCE breaks ties. IEMOCAP official loader
aliases validation to test: DO NOT use that for selection. Instead hold out a
deterministic, label-independent10% of original training-fold conversations
for Adapter validation, removed from Adapter training. The frozen anchor
previously saw these conversations; this is an Adapter-only holdout, NOT a
clean end-to-end validation protocol. Same split for F1/F2. Test fold untouched.
Validation availability is fixed/replayable with the same GCNet algorithm.
No test scores choose adapter checkpoints, loss weight, LR, or thresholds.

Report fixed-anchor F0/F1/F2 paired random8/high3, persistent7, class recall,
CE, corrections/harms, source reconstruction errors, gamma/trajectory/read
diagnostics, cost, and all failures. Random ACC/WF1 drop<.3pp is a protection
target, not a guarantee. If F2 does not beat F1, no unauthorized expansion.

## Verification and execution gate

Tests-first: algebra/strict past/zero histories/padding/independent conversations,
causality/inactive Gap/legacy exact off and zero-on/RNG/gradient two-step/
strict reload and normal/Persistent finite outputs. First-step upstream
Adapter gradients may legitimately be zero due to zero output initialization;
verify upstream gradients after an output update rather than claiming success.
Then real MELD/IEMOCAP GPU forward/backward/optimizer smoke, record max absolute
and Frobenius-relative source errors, frozen hashes, logit parity, peak memory
and wall time. Commit mathematical/real smoke verification BEFORE screening.
Use healthy biggpu GPUs only (physical4 forbidden), immutable sources and
new independent outputs. No existing checkpoints/data are overwritten/deleted.
