# P0 frozen history-effect stratification

**INTERNAL DIAGNOSTIC ONLY.** This experiment uses the retained per-rate
Test-oracle cfg84 no-JEPA Flat checkpoints. It does not train a new model and
must not be reported as a deployable validation-selected result.

## Question and intervention

For the same frozen causal Memory trajectory, the original Flat readout is
replayed as:

\[
P_L=F(L,0,0),\qquad P_{LB}=F(L,B,0),\qquad
P_{LBG}=F(L,B,G).
\]

For every valid utterance, the diagnostic records logits, predictions, the
true-class probability, availability, strictly prior history counts, speaker
identity, speaker switching, and offline label relations. Its paired effects
are

\[
\Delta_{Base}=CE(P_L,y)-CE(P_{LB},y),\qquad
\Delta_{Gap}=CE(P_{LB},y)-CE(P_{LBG},y).
\]

Positive Delta means that the added history evidence lowered cross-entropy.
Corrections and harms denote wrong-to-right and right-to-wrong argmax changes.
Gold label relations are used only after inference for analysis.

## Execution and integrity

- Commit: `31cfe1f` on `feature/ecerc-features`.
- Frozen sources: IEMOCAP-4/6, seeds 66/67/68, five LOSO folds, eight missing
  rates; MELD, seeds 66/67/68, eight missing rates.
- Completed source runs: `33/33`; failed runs: `0`.
- Every run uses one unchanged Memory scan per batch. The three predictions
  differ only at the captured Flat input.
- Replayed full logits are checked against the untouched forward; saved labels,
  predictions and availability are checked against the source artifacts where
  available.
- Remote artifacts:
  `/data2/yb/remote_experiments/osram_history_effect_stratification_20261013/full_31cfe1f`
  and
  `/data2/yb/remote_experiments/osram_history_effect_stratification_20261013/aggregate_31cfe1f`.

The values below first pool the five IEMOCAP folds within each seed, then take
an equal mean over seeds and missing rates. Correction and harm percentages use
all utterances in the corresponding stratum as denominator.

## Overall effects

| Dataset | Base Delta CE | Base correction / harm | Gap Delta CE | Gap correction / harm |
|---|---:|---:|---:|---:|
| IEMOCAP-4 | +0.5075 | 20.53% / 4.28% | +0.0387 | 3.14% / 1.49% |
| IEMOCAP-6 | +0.5723 | 23.83% / 7.26% | +0.0268 | 3.61% / 2.54% |
| MELD | +0.0103 | 0.67% / 0.47% | +0.0030 | 0.18% / 0.17% |

Base is the dominant useful history path on both IEMOCAP tasks. Gap provides a
smaller positive increment. On MELD, both effects are close to zero: the
presence of a trained history vector alone does not reproduce the strong
IEMOCAP dependence.

## Strictly prior history count

### Base Delta CE

| Dataset | 0 | 1 | 2--4 | 5--9 | 10--19 | 20+ |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP-4 | 0.0000 | 0.1413 | 0.2466 | 0.3993 | 0.5617 | 0.6033 |
| IEMOCAP-6 | 0.0000 | 0.1427 | 0.2596 | 0.4145 | 0.5573 | 0.6704 |
| MELD | 0.0000 | 0.0115 | 0.0146 | 0.0081 | 0.0107 | 0.0427* |

`*` MELD has only about 33 utterances per seed in the `20+` stratum; its
non-monotonic small effects do not support a clean history-count trend.

IEMOCAP shows a strong graded relationship: Base becomes increasingly useful
as more prior turns exist. This does not occur merely because first turns have
no history; the increase continues across the nonzero bins.

## Same-speaker prior history

### Base Delta CE

| Dataset | 0 | 1 | 2--4 | 5--9 | 10--19 | 20+ |
|---|---:|---:|---:|---:|---:|---:|
| IEMOCAP-4 | 0.0699 | 0.1839 | 0.4056 | 0.5123 | 0.6228 | 0.6108 |
| IEMOCAP-6 | 0.0496 | 0.2517 | 0.4184 | 0.5423 | 0.6509 | 0.6969 |
| MELD | 0.0027 | 0.0103 | 0.0108 | 0.0203 | 0.0356 | -- |

This is the clearest count relationship. IEMOCAP gains rise substantially with
same-speaker history. MELD moves in the same broad direction but remains much
smaller and has very limited support in its longest bins.

## Speaker continuity and offline label relation

| Dataset | Base Delta: no switch / switch | Base Delta: same-speaker label same / different | Gap Delta: same / different |
|---|---:|---:|---:|
| IEMOCAP-4 | +0.5743 / +0.4835 | +0.6410 / **-0.0739** | +0.0562 / **-0.0453** |
| IEMOCAP-6 | +0.6065 / +0.5742 | +0.7122 / +0.2965 | +0.0391 / +0.0000 |
| MELD | +0.0283 / +0.0062 | +0.0240 / +0.0043 | +0.0071 / +0.0013 |

Speaker switching alone does not erase IEMOCAP Memory benefits, although
MELD benefits much more on non-switch turns. The stronger result is conditional
on the previous same-speaker label. History helps most when the same speaker's
label is unchanged. In IEMOCAP-4, the average Base and Gap CE effects become
negative when that label differs; this is an offline behavioral association,
not an inference rule and not proof that the model detects an emotion shift.

## Decision

This evidence does **not** support another broad parameter sweep. The same Flat
and CHM machinery already uses history strongly in IEMOCAP but barely changes
MELD decisions. The bottleneck is therefore not explained by universally weak
history capacity alone.

The next justified step is P1: keep the current utterance and history budget
fixed while intervening on history identity/content (same-speaker versus
matched other-speaker, correct history versus matched donor/shuffle). This must
test whether the observed benefit depends on the correct causal history rather
than on history-vector presence or magnitude. Only after that intervention can
the second module be placed defensibly in storage, retrieval, or history
organization.

## Validity boundary

- Zeroing trained Flat inputs is distribution shifting; these values describe
  a frozen model's response, not independently trained Local/Base/Gap systems.
- The same utterances recur under eight evaluation masks. Reported values use
  equal seed/rate means and are descriptive, not independent-sample
  significance tests.
- Dataset label spaces, feature pipelines, lengths, and mask protocols differ.
  Cross-dataset contrasts identify hypotheses but do not establish that history
  length or speaker structure is the sole cause.
- Label-relation strata use gold labels offline and cannot be used by a
  deployable gate or inference rule.
