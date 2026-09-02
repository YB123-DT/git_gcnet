# IEMOCAP-6 JEPA contribution ablation

## Question

The current Slot + all-rates Missing-M3 result combines the observed-set model,
mixed-rate training, and a JEPA loss. This experiment isolates whether the
JEPA training signal itself explains the IEMOCAP-6 gain.

The control is accurately described as **JEPA-gradient-off**: it sets
`jepa_weight=0`, while retaining the same forward calls and random-number
consumption. It is not a smaller architecture and it does not delete the
predictor.

## Locked protocol

- dataset: IEMOCAP-6, official fold 5;
- seeds: 66--70, 100 epochs, batch size 32;
- missing rates: 0.0--0.7 in steps of 0.1;
- one all-rates model per seed;
- frozen features: wav2vec-large-c, DeBERTa-large-4, and MANet;
- Slot fusion and Slot representation;
- both temporal and speaker GCNet branches, hidden size 200, window 2/2;
- dual-gate Top-2/4-expert MMoE, latent size 256;
- Adam at `1e-3`, weight decay `1e-5`, gradient clipping 1.0;
- validation selection by the mean W-F1 over all eight rates;
- the selected checkpoint is evaluated on all eight fixed test masks;
- With-JEPA uses `jepa_weight=0.1`; JEPA-gradient-off uses `0.0`;
- existing With-JEPA results are inherited and not retrained.

The pre-registered extension gate required both the eight-rate and high-missing
(0.4--0.7) paired mean deltas to be positive, with at least 3/5 positive seed
aggregates in each group. MOSEI would run only if this gate passed.

## Results

All values are test weighted F1 percentages. Delta is With-JEPA minus
JEPA-gradient-off.

| Miss | With-JEPA mean | JEPA-gradient-off mean | Delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 66.8652 | 65.1607 | +1.7045 | 4/5 |
| 0.1 | 66.7658 | 64.4981 | +2.2677 | 5/5 |
| 0.2 | 65.6024 | 63.6510 | +1.9514 | 5/5 |
| 0.3 | 65.7438 | 64.0848 | +1.6590 | 5/5 |
| 0.4 | 63.1030 | 63.1446 | -0.0416 | 3/5 |
| 0.5 | 62.9960 | 62.0902 | +0.9058 | 3/5 |
| 0.6 | 61.0125 | 61.8495 | -0.8370 | 2/5 |
| 0.7 | 60.7855 | 60.7858 | -0.0003 | 2/5 |

Aggregate results:

| Aggregate | With-JEPA | JEPA-gradient-off | Delta | Positive seeds |
|---|---:|---:|---:|---:|
| Miss 0 | 66.8652 | 65.1607 | +1.7045 | 4/5 |
| Eight-rate mean | 64.1093 | 63.1581 | +0.9512 | 4/5 |
| High missing 0.4--0.7 | 61.9742 | 61.9675 | +0.0067 | 2/5 |

Per-seed aggregate deltas:

| Seed | Eight-rate delta | High-missing delta | Miss-0 delta |
|---:|---:|---:|---:|
| 66 | +0.6828 | -0.2424 | +2.0827 |
| 67 | +1.9211 | +1.7829 | +1.9275 |
| 68 | +1.7448 | +0.7285 | +2.6809 |
| 69 | -0.2728 | -0.9099 | -0.5862 |
| 70 | +0.6800 | -1.3255 | +2.4174 |

Best epochs were `65,61,68,73,76` With-JEPA and
`72,89,79,91,61` with the JEPA gradient disabled.

## Integrity audit

- 5/5 jobs completed with zero runner failures;
- 5/5 histories contain 100 finite epochs;
- 5/5 stored best epochs equal an independent validation argmax;
- after filling five later-added default config fields in the inherited run,
  all 5/5 paired configs differ only in `jepa_weight`;
- model source SHA256 is byte-identical between the two runs;
- parameter count, trainable parameter count, and EMA step count match for 5/5
  seed pairs;
- 40/40 stored mask hashes, label arrays, and availability arrays match between
  the paired arms;
- 80/80 NPZ files independently reproduce stored W-F1, macro-F1, and accuracy,
  with maximum error `2.22e-16`;
- 80/80 result files contain all six label and prediction classes;
- checkpoints are retained only on the remote host and are not committed.

The inherited run was produced before later optional two-stage trainer branches
were added. Its normalized configuration and default joint-training semantics
match, and `model.py` is byte-identical, but the old run did not archive its
producer source hash. Therefore this report does not claim byte-identical
trainer provenance.

`jepa_weight=0` still computes the predictor loss and EMA update to preserve the
forward/RNG path, but its loss contributes zero gradient. Predictor-only
parameters can still move slightly under Adam weight decay; they are not used
by the emotion classifier or evaluation path. The comparison therefore tests
the JEPA objective's total training effect, including its interaction with the
shared global gradient clipping.

## Decision

**Gate: FAIL.** JEPA improves the overall eight-rate mean by 0.9512 points and
is positive for 4/5 seed aggregates, so it has a real low-to-moderate-missing
regularization effect. However, high missing is an aggregate tie (+0.0067) with
only 2/5 positive seeds. At rate 0.6 it is 0.8370 points worse and at 0.7 it is
effectively identical.

Under the pre-registered rule, the MOSEI JEPA-gradient-off extension is not
launched. The evidence does not support claiming that the current JEPA loss is
the mechanism responsible for robustness at high missing rates.
