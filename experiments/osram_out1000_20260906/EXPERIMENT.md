# H4-Output1000 CMU-MOSI capacity diagnostic

- Branch: `feature/osram-complete`
- Source code commit: `ac7b8295a9d57414d422f8a31ee61bdc96856782`
- Dataset: CMU-MOSI, fold 1, frozen wav2vec/DeBERTa/MANet features
- Seeds: 66, 67, 68, 69, 70
- Training: cyclic 0.0–0.7 rates, 100 epochs, batch size 32
- Selection: one checkpoint per seed by 8-rate-mean Test-oracle; per-rate maxima are an additional diagnostic
- Remote output: `/data2/yb/remote_experiments/osram_out1000_20260906`

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**
