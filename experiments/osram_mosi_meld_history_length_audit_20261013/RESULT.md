# MOSI, MELD, and IEMOCAP history-length audit

**DATASET DIAGNOSTIC ONLY.** This report describes the exact dataset pickles
used by the current experiments. It does not by itself establish that history
length causes the observed Memory-performance difference.

## Sources

- MOSI: `dataset/CMUMOSI/CMUMOSI_features_raw_2way.pkl` in the biggpu OSRAM
  runtime checkout.
- MELD: the ECERC official
  `meld_emotion_semantic_features_roberta.pkl` recorded by the MELD runs.
- IEMOCAP 4-class: `dataset/IEMOCAP/IEMOCAP_features_raw_4way.pkl`.
- IEMOCAP 6-class: `dataset/IEMOCAP/IEMOCAP_features_raw_6way.pkl`.
- Splits are read directly from each pickle. No model inference or training is
  performed.

The IEMOCAP files in this worktree are byte-identical to the files used by the
biggpu runtime checkout. Their SHA256 values are respectively
`072cffa9c528dd0536792de7ac85d5a0efaf25d4a9e3ed01f7e45453547cc9fd` and
`e35143e9f087d436779cccd650cb0327e9caa55178d326330f4a8a914e91e023`.
Current IEMOCAP evaluation uses five-session leave-one-session-out (LOSO), so
the pooled statistics below count each of the 151 dialogues exactly once.

## Conversation length

| Dataset | Split | Dialogues | Utterances | Mean length | Median | P25 | P75 | P90 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSI | train | 52 | 1,284 | 24.692 | 24.0 | 16.75 | 30.25 | 33.9 | 63 |
| MOSI | validation | 10 | 229 | 22.900 | 23.5 | 16.50 | 29.50 | 34.0 | 34 |
| MOSI | test | 31 | 686 | **22.129** | **21.0** | 16.00 | 28.00 | 31.0 | 43 |
| MELD | train | 1,038 | 9,989 | 9.623 | 9.0 | 5.00 | 14.00 | 18.0 | 24 |
| MELD | validation | 114 | 1,109 | 9.728 | 10.0 | 5.00 | 13.00 | 17.0 | 23 |
| MELD | test | 280 | 2,610 | **9.321** | **9.0** | 5.00 | 13.00 | 17.0 | 33 |
| IEMOCAP-4 | pooled LOSO evaluation | 151 | 5,531 | **36.629** | **37.0** | 26.00 | 47.00 | 54.0 | 110 |
| IEMOCAP-6 | pooled LOSO evaluation | 151 | 7,433 | **49.225** | **47.0** | 37.50 | 59.00 | 69.0 | 110 |

On test, a MOSI conversation is about `2.37x` longer on average. The average
number of strictly prior turns available to one evaluated utterance is 11.847
for MOSI and 5.904 for MELD (`2.01x`). First utterances, which have no causal
history at all, account for 4.52% of MOSI test utterances and 10.73% of MELD
test utterances.

IEMOCAP is not MELD-like on this axis. Its average conversation is `3.93x`
(4-class) or `5.28x` (6-class) longer than MELD test. Its average evaluated
utterance has 21.520 or 26.985 strictly prior turns, compared with 5.904 for
MELD. Only 2.73%/2.03% of IEMOCAP-4/6 utterances are history-free first turns.

## Speaker-conditioned history

| Dataset | Split | Speakers/dialogue | Same-speaker prior turns/utterance | Has prior same-speaker turn | Previous same-speaker distance | Adjacent same-speaker |
|---|---|---:|---:|---:|---:|---:|
| MOSI | test | 1.000 | **11.847** | **95.481%** | 1.000 | 100.000% |
| MELD | test | 2.664 | **2.321** | **71.418%** | 2.205 | 24.249% |
| IEMOCAP-4 | pooled LOSO evaluation | 1.993 | **11.248** | **94.558%** | 1.931 | 42.175% |
| IEMOCAP-6 | pooled LOSO evaluation | 2.000 | **13.585** | **95.937%** | 1.974 | 31.049% |

MOSI is a single-speaker sequence in this loader. MELD has multiple speakers;
its same-speaker history per utterance is about `5.10x` smaller than MOSI.
IEMOCAP is also multi-speaker, but its much longer conversations provide
`4.85x` (4-class) or `5.85x` (6-class) as many same-speaker prior turns as
MELD. Thus multi-speaker dialogue alone does not imply a small usable-history
budget.

For descriptive continuity only:

- MOSI adjacent non-neutral utterances have the same binary polarity 65.837%
  of the time on test.
- MELD adjacent utterances have the same seven-class emotion label 41.931% of
  the time on test.
- Within MELD test, adjacent same-speaker pairs keep the same label 57.699% of
  the time, versus 36.884% across a speaker switch.
- IEMOCAP-4 and IEMOCAP-6 adjacent utterances keep the same class label
  73.086% and 55.301% of the time, respectively.

These label relations are not model inputs and the MOSI polarity statistic is
not directly comparable to MELD seven-class equality. They only describe why
raw conversation length may overstate the amount of task-relevant history.
The IEMOCAP-4/6 label statistics are also not directly comparable with each
other because the retained utterance sets and class definitions differ.

## IEMOCAP fold coverage

| Held-out session | 4-class dialogues | 4-class utterances | 4-class mean length | 6-class dialogues | 6-class utterances | 6-class mean length |
|---|---:|---:|---:|---:|---:|---:|
| Ses01 | 28 | 1,085 | 38.750 | 28 | 1,373 | 49.036 |
| Ses02 | 30 | 1,023 | 34.100 | 30 | 1,356 | 45.200 |
| Ses03 | 32 | 1,151 | 35.969 | 32 | 1,569 | 49.031 |
| Ses04 | 30 | 1,031 | 34.367 | 30 | 1,512 | 50.400 |
| Ses05 | 31 | 1,241 | 40.032 | 31 | 1,623 | 52.355 |

No single held-out session creates the pooled IEMOCAP conclusion: every fold
has a mean conversation length far above MELD test's 9.321.

## Interpretation

History length is a plausible contributor to the cross-dataset result:

1. MELD provides roughly half as many prior turns per target utterance as
   MOSI, and only 22%--27% as many as IEMOCAP.
2. It provides only about one fifth as many same-speaker prior turns as MOSI
   or IEMOCAP.
3. More MELD utterances occur at the history-free first position.
4. Speaker switching makes the available history less homogeneous than MOSI.
5. IEMOCAP is a useful counterexample to the claim that all multi-speaker
   corpora necessarily have sparse same-speaker history.

However, these marginal dataset statistics do not prove causation. The direct
test is to stratify the frozen MELD `Full - Local` intervention by strictly
prior-turn count and same-speaker-prior count. If Memory gain rises reliably
with these counts, short/sparse history is supported as a mechanism; if it
does not, history length is not the main bottleneck.
