# Launch evidence

INTERNAL DIAGNOSTIC ONLY; per-rate Test-oracle W-F1.

- Implementation source: `5376948` on `feature/ecerc-features`, pushed to
  `YB123-DT/git_gcnet` before the formal run.
- Server: `biggpu`, physical GPU3,
  `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`, process2596163.
- Persistent tmux: `mosi_recent_1018`; started2026-10-18T03:02:44UTC.
- Immutable helper snapshot:
  `/home/qinjinghui/remote_experiments/osram_mosi_recent_history_20261018/source_5376948`.
- Numerical source: archived `ad211c0`; unchanged CHM and Flat implementation.
- Training output:
  `/home/qinjinghui/remote_experiments/osram_mosi_recent_history_20261018/seed66_v1`.
- Log: same experiment root, `train_seed66_v1.log`.
- One seed66 run, W4,128-dimensional attention,4heads; original100epochs .001
  followed automatically by full-state continuation50epochs .0001.
- Existing experiments were not killed or modified. Physical GPU4 excluded.

## Actual verification

37 CPU tests passed. The local test environment emits an existing optional
torch-sparse/libcusparse warning; the remote training uses the original s0
environment, not this local environment.

The separate real MOSI smoke completed one epoch/two actual optimizer steps:
18.439seconds, peak allocated CUDA memory1,312,117,248bytes. Gradients were
finite; new parameters were included in the optimizer. Same-RNG zero-initial
on/off outputs matched exactly. All eight evaluation mask hashes matched the
original seed66 reference. See `SMOKE_PARITY.json` and `SMOKE.json`.

Smoke scores are not trained-model performance evidence. Formal training is
running; no final improvement is claimed.

## Reproduce

From the helper snapshot above:

```bash
CUDA_VISIBLE_DEVICES=GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a OMP_NUM_THREADS=1 \
  /data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  experiments/osram_mosi_recent_history_20261018/run.py \
  --output /home/qinjinghui/remote_experiments/osram_mosi_recent_history_20261018/seed66_v1 \
  --gpu-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a \
  --wrapper-commit 5376948
```

Do not repeat the launch into an existing output. The runner deliberately rejects
duplicate initialization. For a new authorized reproduction use a new output.
Each stage preserves full recovery state and per-rate BEST weights/predictions;
stage150 restores model, Adam, RNG, masks, selection history and BEST references.
