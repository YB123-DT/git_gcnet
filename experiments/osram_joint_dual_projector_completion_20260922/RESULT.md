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
| 0.0 | 86.856±0.287 | 87.121±0.516 | +0.266 | 2/3 |
| 0.1 | 85.108±0.379 | 84.871±0.394 | -0.237 | 1/3 |
| 0.2 | 81.934±1.432 | 82.301±0.522 | +0.367 | 2/3 |
| 0.3 | 80.423±1.391 | 79.364±1.710 | -1.059 | 1/3 |
| 0.4 | 78.446±2.142 | 76.842±2.644 | -1.604 | 0/3 |
| 0.5 | 76.141±1.643 | 74.986±3.489 | -1.155 | 1/3 |
| 0.6 | 75.973±0.985 | 73.992±1.945 | -1.981 | 0/3 |
| 0.7 | 73.391±3.852 | 71.256±3.597 | -2.134 | 1/3 |

- Eight-rate mean: **79.784% → 78.842% (-0.942 pp)**.
- High-missing mean (0.5/0.6/0.7): **75.168% → 73.412% (-1.757 pp)**.
- Positive paired seed-rate cells: **8/24**.

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
- Evaluation mask hashes equal the matched cfg84 control for all seeds/rates.
- A unit test changes every unavailable raw modality block and observes
  bit-identical logits, ruling out missing-feature leakage.

**No-Go.** Preserving the ordinary online projectors fixes the catastrophic
miss=0 failure of the earlier fully frozen representation, but frozen predicted
latents still reduce overall and high-missing performance. In particular,
T-missing remains around the previous low-60% platform rather than improving.

