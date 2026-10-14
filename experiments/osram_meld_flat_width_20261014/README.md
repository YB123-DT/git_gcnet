# MELD Flat adapter capacity screen

INTERNAL DIAGNOSTIC ONLY. Three runs, seed66, 50 epochs, original GCNet
`gcnet-random-mask-v1` host/guest batch sampler, rates0.0–0.7 cyclic. Adam
lr1e-3, batch32 and all other source settings remain identical. Training uses
historical `b4a3138` code. No backbone implementation change is needed.

Only `osram_adapter_hidden_dim` changes to 512/1024/2048. The original
1600-width result is reused from the immediately preceding reproduction:
`/data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline`.

Flat's adapter is `4352 -> width -> 1600`; output dimension, Local skip,
classifier, Memory heads/dimensions, queries, writes and loss are unchanged.
Existing initialization isolation preserves all other modules' random draws.
This is hidden-width tuning, not an output_dim sweep or Memory capacity sweep.

Run each width from the immutable current runner worktree on biggpu GPU0,
three parallel runs, each with its own output and log. Healthy GPU UUID:
`GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45`. Host GPU4 is forbidden.

```bash
CUDA_VISIBLE_DEVICES=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 \
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m \
experiments.osram_meld_flat_width_20261014.run \
  --width 512 \
  --original-code /data2/yb/remote_experiments/osram_meld_write_retention_20261014/training_b4a3138 \
  --source-config /data2/yb/paper/remote_runs_20261012/results/osram_gcnet_random_mask_iemocap_meld_20261013_r1/meld/meld_flat_s66/config.json \
  --baseline-reference /data2/yb/remote_experiments/osram_meld_write_retention_20261014/seed66_reproduced_b4a3138/baseline \
  --meld-pickle /data2/yb/paper/remote_runs_20261012/data/meld/meld_emotion_semantic_features_roberta.pkl \
  --output /data2/yb/remote_experiments/osram_meld_flat_width_20261014/width512 \
  --gpu-uuid GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
```

The runner validates that the sole effective config difference is hidden
width. Every run retains the eight per-rate best checkpoints and saved masks.
Report per-rate ACC/W-F1 and mean/high mean under established selection;
also compute seven-class UA from final saved predictions on paired masks.
Do not conflate that UA with an unavailable selection-epoch UA. Availability
artifacts must match the reference before reporting paired prediction deltas.
