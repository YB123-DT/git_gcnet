# P1 matched history-content intervention

**INTERNAL DIAGNOSTIC ONLY.** The source checkpoints retain their historical
per-rate Test-oracle selection. This experiment is frozen inference: it does not
train or modify a model and is not a deployable performance result.

## Protocol and completion

- Implementation/source commit: `4bd9911`.
- Frozen cfg84 no-JEPA causal Flat checkpoints: IEMOCAP-4, IEMOCAP-6 and MELD.
- Seeds 66/67/68; five IEMOCAP folds per seed and one MELD fold per seed.
- Missing rates 0.0--0.7; 33/33 sources completed, with no failed source.
- For each target and modality, the latest observed strict-past utterance from
  the target speaker is held fixed. Only that modality slice is replaced by a
  matched strict-past donor from either the same or a different speaker.
- Selection uses availability, speaker ID and position only, never labels or
  predictions. Current/future inputs, availability trajectory and current
  Local remain unchanged (Local passed the configured numerical tolerance).
- Raw aggregate:
  `/data2/yb/remote_experiments/osram_history_content_intervention_20261014/aggregate_4bd9911`.

`Delta CE = CE(replaced) - CE(original)`; positive values mean that replacing
the real historical content harmed the frozen model. Values below pool eligible
paired rows over all three seeds and, unless marked high missing, all eight
missing rates. Correction/harm are original-wrong to replaced-correct and
original-correct to replaced-wrong rates among eligible rows.

## All-rate pooled results

| Dataset | Historical modality | Same-speaker donor Delta CE | Other-speaker donor Delta CE | Same correction / harm | Other correction / harm |
| --- | --- | ---: | ---: | ---: | ---: |
| IEMOCAP-4 | A | 0.00766 | 0.01226 | 0.96% / 1.38% | 0.99% / 1.59% |
| IEMOCAP-4 | T | **0.05287** | **0.08206** | 2.88% / 4.72% | 3.09% / 6.05% |
| IEMOCAP-4 | V | 0.00878 | 0.02853 | 0.99% / 1.34% | 1.90% / 2.67% |
| IEMOCAP-6 | A | 0.00995 | 0.01803 | 1.63% / 2.10% | 1.69% / 2.46% |
| IEMOCAP-6 | T | **0.04819** | **0.06477** | 4.85% / 6.63% | 5.22% / 7.89% |
| IEMOCAP-6 | V | 0.00737 | 0.01213 | 1.28% / 1.65% | 2.53% / 3.00% |
| MELD | A | 0.00001 | 0.00003 | 0.01% / 0.01% | 0.01% / 0.01% |
| MELD | T | **0.00485** | **0.00732** | 0.52% / 0.58% | 0.42% / 0.50% |
| MELD | V | -0.00001 | -0.00003 | 0.02% / 0.01% | 0.02% / 0.00% |

At high missing rates (0.5/0.6/0.7), Text Delta CE was 0.06853/0.10080
(same/other speaker) for IEMOCAP-4 and 0.06421/0.07891 for IEMOCAP-6. MELD
remained much less sensitive at 0.00733/0.00758.

## Separating Text content from speaker metadata

The completed design already contains the proposed Text-only and
same-speaker/wrong-Text analyses:

- The Text intervention changes only the strict-past Text feature slice;
  historical Audio/Visual, availability and every speaker ID remain unchanged.
- In the `same_speaker` arm, the replacement Text comes from another utterance
  by the target speaker. Thus explicit speaker metadata and speaker relation are
  preserved while Text content changes.

The sizeable same-speaker Text effects (0.05287 for IEMOCAP-4 and 0.04819 for
IEMOCAP-6; 0.06853 and 0.06421 at high missing rates) establish that speaker ID
alone is insufficient: the model is sensitive to the speaker-associated Text
content. Using an other-speaker Text donor adds 0.02918 and 0.01658 Delta CE
respectively over the same-speaker donor when pooled across rates. This extra
effect is compatible with speaker-conditioned content, style or trajectory
mismatch, but it does **not** isolate the causal contribution of explicit
speaker ID because the intervention never changes `qmask`.

## Supported conclusions

1. IEMOCAP decisions depend on the identity of historical content, rather than
   merely on the presence or dimensionality of a history vector. Replacing one
   observed historical slice while preserving the full availability trajectory
   consistently increases CE.
2. Historical Text is the dominant content-sensitive modality in both IEMOCAP
   label settings. The effect increases under high missingness.
3. Other-speaker replacement is generally more harmful than same-speaker
   content shuffling in IEMOCAP, particularly for Text and IEMOCAP-4 Visual.
   This supports studying speaker-conditioned history organization, but does
   not isolate speaker identity from the recorded donor-lag mismatch.
4. MELD is almost insensitive to Audio/Visual replacement and only weakly
   sensitive to Text replacement. A mechanism justified by IEMOCAP history
   dependence must therefore not be claimed as a universal explanation for
   MELD.
5. Harm exceeds correction for the material IEMOCAP interventions. The result
   is consistent with useful real history being disrupted, not with arbitrary
   perturbation routinely repairing the classifier.

This diagnostic does not establish natural-language causality, recover missing
modalities, or prove that a particular new module will improve task metrics.
Correlated utterances are not independent significance samples, and the two
donor arms are not exact-lag matched.
