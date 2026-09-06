# Experiment record

## Run identity

- Local experiment directory: `experiments/osram_text_anchor_20260906/`
- Remote output root: `/data2/yb/remote_experiments/osram_text_anchor_20260906_retry/`
- Remote host: `biggpu`
- GPUs: 0, 1, 2, 3, 7 (GPU4 was not used).
- Source snapshot hashes matched the local `feature/osram-complete` source for
  `gcnet_missing_m3/model.py`, `train_gcnet.py`, and `osram.py`.
- Fixed mask SHA256 values match the OSRAM+mean control for all five seeds and
  all eight rates.

## Run details

Each seed ran 100 epochs and wrote:

- `best.pt`
- `config.json`
- `history.json`
- `metrics.json`
- `diagnostics.json`
- eight `predictions_miss_*.npz` files

All five runs completed without a traceback, NaN, or non-finite output. The
first launcher attempt exited immediately because it omitted the remote
working-directory change and could not import `gcnet_missing_m3`; it did not
run an epoch and is not included in the result files. The corrected launch
completed all five runs.

## Selection semantics

The `metrics.json` result is the one-checkpoint-per-seed selection: the epoch
maximizing the mean Test weighted-F1 over all eight rates. `per_seed_rate.csv`
also includes a separate per-rate Test-oracle diagnostic, where each rate may
choose a different epoch. The latter is useful for mechanism diagnosis only.
