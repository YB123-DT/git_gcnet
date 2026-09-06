# V64 CMU-MOSI capacity diagnostic

- Branch: `feature/osram-complete`
- Source code commit: `ac7b8295a9d57414d422f8a31ee61bdc96856782`
- Dataset: CMU-MOSI, fold 1, official feature root
- Seeds: 66, 67, 68, 69, 70
- Training: cyclic mixed-rate schedule, rates 0.0–0.7, 100 epochs, batch size 32
- Fixed settings: latent=256, hidden=200, H4 control otherwise unchanged, fusion=mean, structured predictor, JEPA joint objective
- Selection: one checkpoint per seed by 8-rate-mean Test-oracle; `per_seed_rate.csv` additionally reports the independent per-rate Test-oracle maxima for diagnosis
- Remote output: `/data2/yb/remote_experiments/osram_v64_20260906`

Seed 70 in the KV64 group initially hit a transient CUDA initialization failure; it was rerun once in the documented retry directory and only the successful retry is included. No other run was retrained after completion.

**INTERNAL DIAGNOSTIC ONLY — NOT A FORMAL PAPER RESULT.**
