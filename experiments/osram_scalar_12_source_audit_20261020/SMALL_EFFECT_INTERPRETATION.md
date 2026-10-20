# Why small12-slot removals do not show that Memory is unused

INTERNAL DIAGNOSTIC ONLY. Additional read-only analysis of already saved
per-condition JSONs from the completed audit. No new inference or training.
Same seed66 fixed miss=.3 checkpoint and exact masks as the12-slot tables.

Here full means F(Local,Base,Gap); all history zero means F(Local,0,0).
The model,Local,Memory scan,queries and task head stay fixed. This is not a
separately trained Local-only model, and not removal of historical raw inputs.
The zero-history result is the already computed none coalition, not a new run.

| Dataset | Rate | Full W-F1 | All history zero W-F1 | Zero − Full pp |
|---|---:|---:|---:|---:|
| CMUMOSI | 0.0 | 86.2270 | 86.1378 | -0.0892 |
| CMUMOSI | 0.1 | 83.3347 | 83.2683 | -0.0664 |
| CMUMOSI | 0.2 | 81.7469 | 80.3664 | -1.3805 |
| CMUMOSI | 0.3 | 82.1220 | 79.0319 | -3.0902 |
| CMUMOSI | 0.4 | 77.0180 | 76.8185 | -0.1995 |
| CMUMOSI | 0.5 | 74.0977 | 70.1494 | -3.9483 |
| CMUMOSI | 0.6 | 72.4280 | 66.9044 | -5.5237 |
| CMUMOSI | 0.7 | 67.0113 | 63.8775 | -3.1338 |
| CMUMOSEI | 0.0 | 86.9468 | 85.6420 | -1.3048 |
| CMUMOSEI | 0.1 | 85.8408 | 83.2984 | -2.5425 |
| CMUMOSEI | 0.2 | 85.2688 | 80.9677 | -4.3011 |
| CMUMOSEI | 0.3 | 84.5401 | 77.9442 | -6.5959 |
| CMUMOSEI | 0.4 | 82.3938 | 74.5567 | -7.8372 |
| CMUMOSEI | 0.5 | 81.4720 | 71.6020 | -9.8700 |
| CMUMOSEI | 0.6 | 78.9221 | 68.2728 | -10.6494 |
| CMUMOSEI | 0.7 | 77.6210 | 63.5403 | -14.0808 |

Eight-rate means: MOSI77.9982 ->75.8193 W-F1 (-2.1789pp);
MOSEI82.8757 ->75.7280 W-F1 (-7.1477pp).

Small individual effects and larger all-history effects coexist. This supports
joint dependence on the readout evidence, not proof that any one source is
necessary or that redundancy alone explains it. Nonlinear Flat interactions,
multiple related read paths, and off-distribution zeroing are not separated.
A12-cell effect is a conditional marginal intervention, not a standalone source
classifier score or semantic causal attribution. Slot effects must not be added.

An unchanged aggregate score also need not mean unchanged predictions:
MOSI rate0.0 removing Base<-T has7corrections and7harms among656nonzero-label
utterances, so ACC stays86.2805 while14polarity predictions change.
MOSI rate0.7 removing Gap-T<-T gives11corrections versus22harms among447,
ACC59.2841 ->56.8233; MOSEI rate0.7 gives61corrections versus105harms among2447,
ACC74.1725 ->72.3743, W-F1 73.4781 ->70.6737.

No statistical-significance or new-module effectiveness claim. All conclusions
refer to these frozen checkpoints; all-history zeroing may be distribution-shifting.
Raw extraction and full/zero ACC,W-F1,MSE are in ZERO_HISTORY_COMPARISON.json.
