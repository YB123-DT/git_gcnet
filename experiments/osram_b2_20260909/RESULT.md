# B2 MOSI completed results

INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.

Five seeds (66–70) completed Stage1 100 epochs and Stage2 100 epochs. This experiment uses **8-rate-mean Test-oracle checkpoint selection**: one selected checkpoint per seed reports all eight rates, not a separate epoch per rate. Stage1 audit uses the fixed final predictor and test miss=0.5 without checkpoint selection.

P0 is the inherited causal OSRAM write-step=0.6 checkpoint. B2 initializes from that checkpoint, adds source-only pretraining and 100 epochs of joint fine-tuning. Thus this is **not an equal-training-budget backbone ablation**. Both use cyclic training and matching evaluation masks. No new training or controls were launched for this report.

## Weighted F1 (%)

| Missing rate | P0 | B2 | Delta (pp) |
|---|---:|---:|---:|
| 0.0 | 87.1899 | 86.4524 | -0.7375 |
| 0.1 | 85.0365 | 84.4610 | -0.5755 |
| 0.2 | 81.9009 | 81.7059 | -0.1950 |
| 0.3 | 80.3681 | 80.2492 | -0.1189 |
| 0.4 | 77.6618 | 77.1592 | -0.5026 |
| 0.5 | 74.9547 | 75.2723 | +0.3177 |
| 0.6 | 74.3858 | 74.2859 | -0.1000 |
| 0.7 | 74.0283 | 73.0315 | -0.9968 |
| Eight-rate mean | 79.4407 | 79.0772 | -0.3636 |
| High missing (0.5/0.6/0.7) | 74.4563 | 74.1966 | -0.2597 |

Earlier P0 80.315% was the **five-rate** mean (0/.1/.3/.5/.7); it must not be compared with this eight-rate B2 mean.

| Seed | P0 epoch | B2 epoch | P0 eight-rate mean | B2 eight-rate mean | Delta (pp) |
|---|---:|---:|---:|---:|---:|
| 66 | 30 | 17 | 79.6533 | 80.2137 | +0.5604 |
| 67 | 41 | 32 | 79.7766 | 78.7312 | -1.0455 |
| 68 | 47 | 8 | 79.1404 | 78.3096 | -0.8307 |
| 69 | 46 | 7 | 79.9120 | 79.1458 | -0.7662 |
| 70 | 53 | 26 | 78.7214 | 78.9856 | +0.2642 |

Only 2/5 seeds improve overall; only 1/8 rate means improves. This version does not demonstrate an overall benefit. No statistical significance or mechanism causality is claimed.

## Stage1 regression prediction audit

Five-seed means; these are the actual regression predictions used for completion, not contrastive-head outputs. Retrieval and chance below are fractions. Effective rank uses the existing audit's centered singular-value entropy.

| Direction | Raw cosine | Centered cosine | Real-minus-shuffle cosine | Retrieval | Chance | Prediction rank | Teacher rank | Std ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A→T | 0.93131 | 0.02568 | 0.000387 | 0.00368 | 0.00847 | 16.355 | 76.374 | 0.07899 |
| V→T | 0.92798 | 0.02653 | 0.000390 | 0.00875 | 0.00858 | 19.040 | 76.712 | 0.12698 |
| AV→T | 0.93036 | 0.04658 | 0.000748 | 0.01188 | 0.01196 | 19.967 | 60.115 | 0.08266 |

The high raw cosine is not evidence of strong sample-specific completion: centered cosine and real/shuffle separation remain small, and retrieval is near or below chance. These are **Stage1** measurements, not an audit of the final Stage2 predictor; they cannot alone establish the cause of Stage2 performance loss.

## Provenance

- Branch: `feature/osram-complete`; implementation `d7d02d1`, launch `a317453`.
- Remote B2: `/data2/yb/remote_experiments/osram_b2_20260909/formal/seed_{66..70}/`.
- Remote P0: `/data2/yb/remote_experiments/osram_write_step_train_20260909/mosi/seed_{66..70}/`.
- Copied original metrics and Stage1 audits: `results/seed_*/`.
- Existing implementation verification is recorded in `VERIFICATION.md`; this report does not rerun tests or alter model code.
- Checkpoints and sample-level predictions remain on the experiment server; they are not uploaded in this report.
