# CMU-MOSI bottleneck diagnosis

This note separates the observed weaknesses instead of treating every
negative ablation as evidence against the whole model.

## 1. What the direct MMoE ablation says

The Full-4E OSRAM reference uses the current four-expert, top-2 dual-gate
MMoE. The Single-1E control uses one expert and top-1 routing while keeping
the predictor interface and all other settings fixed.

- Per-rate Test-oracle extraction: `80.781% -> 80.851%` (+0.070 points).
- High missing mean (0.5/0.6/0.7): `76.586% -> 76.677%` (+0.091 points).
- Strict one-checkpoint eight-rate mean: `80.267% -> 80.267%` (+0.001
  points).

Therefore the routing mechanism is not a stable source of the current error.
The single-expert control is not a parameter-matched replacement, so this is
not a final architecture recommendation; it is enough to deprioritize expert
routing as the first bottleneck.

## 2. Structural forward-path audit

In `gcnet_missing_m3/model.py`, the predictor is computed only when
`predict_missing=True` or `classification_completion=True`. The formal
configuration uses `classification_completion=false`. The training loop calls
`predict_missing=train_jepa`, while `evaluate_rate()` calls
`predict_missing=False` and asserts that no predictions are returned.

Thus the formal test path is:

```text
incomplete features -> ObservedSet/OSRAM -> emotion head
```

The predicted missing latent is not written into the classification hidden.
The predictor can only affect the classifier indirectly through JEPA gradients
to shared Student/OSRAM parameters during training.

## 3. One-batch gradient audit

The audit loads the completed Single-1E seed-66 checkpoint and evaluates one
cyclic training batch at rate 0.5. It is a mechanism audit, not a benchmark
selection step.

| Loss | smax_fc | OSRAM | ObservedSet | MissingPredictor |
|---|---:|---:|---:|---:|
| classification | 6.3195 | 0.6296 | 0.5984 | 0 |
| raw JEPA | 0 | 0.0446 | 0.0747 | 0.1520 |

With `jepa_weight=0.1`, the JEPA contribution reaching OSRAM is about 0.7%
of the classification gradient on this batch. This explains why changing the
MMoE routing has almost no classification effect without claiming that the
predictor is mathematically useless.

## 4. Latent target audit already completed

The prior five-seed latent audit found regression completion prototype
collapse at miss=0.7:

- regression effective rank: 7.15--9.38 / 256;
- retrieval: 0.256--0.304%, close to chance;
- centered Real-vs-Shuffle cosine gap: 0.00032--0.00146;
- teacher rank: 18.83--47.79 / 256.

The teacher is not constant; the actual regression prediction is the part that
collapses. The independent contrastive head contains slightly more
sample-level information, but it is not the head used for completion.

## 5. Why completion did not repair classification

Two existing controls isolate the issue:

1. Connecting InfoNCE to the actual regression prediction reduced the
   prototype shortcut, but decreased five-seed eight-rate emotion W-F1 by
   0.805 points.
2. Directly feeding predicted latent residuals into the emotion hidden
   decreased the eight-rate mean by 0.478 points.

So the current problem is not simply “the MMoE needs more experts”. It is an
objective/representation mismatch: a target-modality latent that is useful for
cross-modal prediction is not automatically an emotion-useful replacement for
the missing input.

## Current working diagnosis

```text
OSRAM context path       : demonstrably useful
MMoE routing              : no stable contribution
regression completion     : prototype-collapsed
test-time completion      : harmful when enabled
JEPA -> emotion coupling  : weak and not yet isolated for OSRAM
```

The next experiment is therefore `emotion-only` versus the current OSRAM
joint objective, with the same masks and five seeds. It tests whether the
remaining JEPA gradient helps or harms the emotion backbone before any new
predictor architecture is designed.

This directory is an internal diagnostic record, not a formal benchmark
result.
