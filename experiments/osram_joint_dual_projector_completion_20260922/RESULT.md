# Dual-projector frozen completion on MOSI

> Internal diagnostic only. Each seed × missing rate uses its own best Test
> W-F1 epoch, matching the existing cfg84 diagnostic protocol.

## Design actually tested

- The ordinary cfg84 online projectors, OSRAM, and emotion classifier are
  initialized normally and trained end to end.
- A second projector bank and `SourceOnlyM3Predictor` are loaded from the
  three-dataset utterance JEPA checkpoint and remain frozen.
- The frozen branch projects only genuinely observed raw slots. It predicts
  missing latents and feeds them through `CompletedReadFusion` into the current
  OSRAM read node.
- Frozen predictions never enter the persistent OSRAM write, and complete raw
  missing features are never read by the classification path.

The implementation therefore tests predicted completion, not oracle feature
insertion.

## Three-seed result

| Missing rate | Matched cfg84 no-JEPA | Dual projector | Paired delta | Positive seeds |
|---:|---:|---:|---:|---:|
| 0.0 | 88.419±0.276 | 87.121±0.516 | -1.298 | 0/3 |
| 0.1 | 85.845±0.821 | 84.871±0.394 | -0.974 | 0/3 |
| 0.2 | 83.431±1.336 | 82.301±0.522 | -1.131 | 0/3 |
| 0.3 | 80.999±1.071 | 79.364±1.710 | -1.636 | 0/3 |
| 0.4 | 78.996±1.857 | 76.842±2.644 | -2.154 | 0/3 |
| 0.5 | 77.327±0.568 | 74.986±3.489 | -2.340 | 0/3 |
| 0.6 | 75.848±0.345 | 73.992±1.945 | -1.857 | 0/3 |
| 0.7 | 73.607±3.723 | 71.256±3.597 | -2.350 | 0/3 |

- Eight-rate mean: **80.559% → 78.842% (-1.718 pp)** for the
  seed-66/67/68 paired comparison. The separately reported five-seed original
  cfg84 mean is 80.445%.
- High-missing mean (0.5/0.6/0.7): **75.594% → 73.412% (-2.182 pp)**.
- Positive paired seed-rate cells: **0/24**.

Per-seed dual-projector eight-rate means are 80.027%, 77.251%, and 79.247%
for seeds 66, 67, and 68 respectively.

## Pattern diagnostic

These are absolute dual-projector scores. `cell mean` averages available
seed-rate pattern cells; `sample pooled` pools the corresponding real samples.

| Observed pattern | Cell mean W-F1 | Sample-pooled W-F1 |
|---|---:|---:|
| A | 61.555 | 62.143 |
| T | 84.123 | 84.916 |
| V | 64.117 | 63.604 |
| AT | 86.238 | 85.919 |
| AV | 64.314 | 65.419 |
| TV | 86.769 | 86.897 |
| ATV | 86.384 | 86.323 |
| T-missing (A/V/AV) | 64.282 | 63.732 |
| T-present (T/AT/TV/ATV) | 86.015 | 86.134 |

## Integrity and decision

- Frozen completion parameters: 1,785,352; their SHA-256 is unchanged before
  and after training for all three seeds.
- Trainable downstream/online parameters: 13,776,045.
- Evaluation mask hashes equal the established mask reference for all seeds/rates.
- A unit test changes every unavailable raw modality block and observes
  bit-identical logits, ruling out missing-feature leakage.

**No-Go.** Preserving the ordinary online projectors fixes the catastrophic
miss=0 failure of the earlier fully frozen representation, but frozen predicted
latents still reduce overall and high-missing performance. In particular,
T-missing remains around the previous low-60% platform rather than improving.
