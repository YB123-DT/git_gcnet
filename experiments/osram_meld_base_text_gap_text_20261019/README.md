# MELD Base←T / Gap-T←T paired four-condition audit

INTERNAL DIAGNOSTIC ONLY. No training or new module. Reuse the original protected
seed66 Flat miss0.3 checkpoint, lr1e-5/wd1e-5, same as the completed12-source audit.
Replay its exact original GCNet masks at random0.0--0.7. Do not switch anchors.

Select valid utterances with current Text missing AND strictly-past Text ever
seen. All four conditions use the exact same selected utterances per rate:

1. Original, retain Base←T and Gap-T←T.
2. Subtract only Base←T at the final Flat input.
3. Subtract only Gap-T←T at the final Flat input.
4. Subtract both, leaving every other source/row and Local Skip unchanged.

Cache already has1--3, but not4 or the full Local/Flat input. One frozen Memory
scan per batch is therefore necessary to compute4; reproduce cached1--3 on the
selected cohort exactly, validate cached NPZ hashes/IDs/labels/speakers/Mask and
source provenance. No modifications to Memory writes, Query, input availability
or classifier. Capture uses the existing opt-in source tracker, TASR off.

Report ACC(primary), UA, W-F1, CE, paired corrections/harms, conditional effects:
Gap contribution with Base←T present versus absent, and converse. CE finite
difference is CE_both_off - CE_base_off - CE_gap_off + CE_full, not a semantic
redundancy score. Nonlinear output differences need not sum. No gold-driven
switching, threshold search, extra seed, Persistent training or optimizer.

Random0.0 has no eligible targets and is N/A, not a zero-performance row. Fully
Persistent patterns also have no eligible targets: Text is either always present
or never observed. Do not mix those empty conditions into a random mean. If
reporting a macro it is over the explicitly named seven nonempty rates, not the
original8-rate benchmark. Whole-dataset baseline numbers do not apply to this
selected Text-missing/history-Text-seen cohort.

Run on biggpu/cmerc-repro, healthy GPU UUID, independent immutable source/output.
Keep all old protected checkpoints and artifacts. Frozen input removal may be
OOD; results describe this fixed classifier, not information absence or a proved
new method. Every pair is matched within a rate; different rates need not select
the same target utterances.
