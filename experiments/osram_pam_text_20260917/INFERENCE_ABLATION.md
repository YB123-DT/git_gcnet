# PAM-T inference ablation: Normal / Zero / Shuffle / Oracle-Fusion

**Internal diagnostic only; not a formal paper result.**

Same PAM-T checkpoints, same test masks, evaluation only.

- Normal: original PAM prediction
- Zero: `z_hat_text` set to zero
- Shuffle: `z_hat_text` shuffled among T-missing samples
- Oracle-Fusion: real Student Text latent passed through the same
  `CompletedReadFusion` interface

## Macro W-F1 (%)

| Rate group | Normal | Zero | Shuffle | Oracle-Fusion |
|---|---:|---:|---:|---:|
| all 8 rates | 80.1162 | 79.1502 | 79.1996 | 82.1841 |
| nonzero rates 0.1–0.7 | **79.0871** | **77.9831** | **78.0395** | **81.4504** |

## Per-rate W-F1 (%)

| Rate | Normal | Zero | Shuffle | Oracle-Fusion | N−Z | N−S | O−N |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 87.3198 | 87.3198 | 87.3198 | 87.3198 | +0.0000 | +0.0000 | +0.0000 |
| 0.1 | 85.0395 | 84.6508 | 84.7606 | 85.6438 | +0.3887 | +0.2789 | +0.6043 |
| 0.2 | 82.4686 | 81.3730 | 81.5224 | 84.1888 | +1.0955 | +0.9462 | +1.7202 |
| 0.3 | 81.2851 | 80.5765 | 80.0848 | 82.8526 | +0.7086 | +1.2003 | +1.5674 |
| 0.4 | 78.4480 | 77.2656 | 77.7248 | 80.4146 | +1.1825 | +0.7233 | +1.9666 |
| 0.5 | 76.5535 | 74.9587 | 75.0846 | 80.0288 | +1.5948 | +1.4689 | +3.4753 |
| 0.6 | 75.5602 | 74.5863 | 74.3195 | 79.1374 | +0.9739 | +1.2407 | +3.5772 |
| 0.7 | 74.2550 | 72.4705 | 72.7798 | 77.8871 | +1.7845 | +1.4752 | +3.6321 |

Nonzero-rate averages:

- Normal − Zero = **+1.1041 pp**
- Normal − Shuffle = **+1.0476 pp**
- Oracle − Normal = **+2.3633 pp**

High missing rates 0.5–0.7:

- Normal − Zero ≈ **+1.45 pp**
- Normal − Shuffle ≈ **+1.40 pp**
- Oracle − Normal ≈ **+3.56 pp**

## Additional numbers

- prior-write coverage, weighted by T-missing count: **85.52%**
- `|s_normal − s_zero|` mean: **0.9988 pp** (all rates), **1.1041 pp** (nonzero rates)

## Direct interpretation

| Result | Meaning |
|---|---|
| Normal > Zero | PAM prediction is used by the completion/read path; it is not ignored. |
| Normal > Shuffle | PAM output carries some sample-specific correspondence, not just a generic vector or scale. |
| Oracle ≫ Normal | The fusion interface can exploit real Text latent much better; PAM prediction quality is the bottleneck. |
| Oracle − Normal grows with missing rate | The harder the missing setting, the larger the headroom of real Text latent. |
| prior-write coverage ≈ 85.5% | Most T-missing utterances do have a prior valid write; low coverage is not the main problem. |

## Answer to the original question

The result is closest to:

$$ \boxed{\text{fusion 能用 Text，但 PAM 没预测好}} $$

More precisely:

- completion residual path is used: Normal > Zero;
- it is not only a scale/regularization effect: Normal > Shuffle;
- the large gap to Oracle-Fusion shows that the prediction quality of `z_hat_text`, not the read path or adapter gate, is the limiting factor;
- prior-write coverage is high, so “no history association to read” is not the dominant failure mode.

## Note on the three initial suspicions

1. **Low prior-write coverage**: not supported. Weighted coverage is 85.5%.
2. **PAM loss only supervises missing positions**: not true in the implementation.
   The PAM regression mask is every valid position with at least one visible A/V
   source, including T-present positions through read-before-write.
3. **Predicted Text is ignored by the zero-init residual adapter**: not supported.
   The adapter is trained and used: Normal > Zero/Shuffle, and Oracle-Fusion
   through the same interface is much better.

The remaining bottleneck is PAM prediction quality itself.
