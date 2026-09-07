# MMoE routing ablation: CMU-MOSI

This run is an **internal diagnostic only**. It is intended to locate a weak
point in the current model, not to claim a formal benchmark result.

## Conditions

- `Full-4E`: inherited full OSRAM with the current dual-gate MMoE
  (`num_experts=4`, `top_k=2`) from
  `experiments/osram_heads8_out700_20260906/`.
- `Single-1E`: newly trained full OSRAM with
  `num_experts=1`, `top_k=1`.

The single-expert setting is the smallest direct test of whether expert
routing itself is useful. All other settings are held fixed.

## Protocol

- CMU-MOSI, fold 1
- Seeds 66--70
- Cyclic mixed-rate training over 0.0--0.7
- 100 epochs, batch size 32, LR 1e-3
- Frozen wav2vec-large-c-UTT, deberta-large-4-UTT and manet_UTT features
- Eight-rate-mean Test-oracle selection for one checkpoint per seed

## Interpretation rule

- `Single-1E < Full-4E`: routing/expert specialization is contributing signal;
  the weak point should be sought elsewhere before changing MMoE.
- `Single-1E ~= Full-4E` or `Single-1E > Full-4E`: expert competition is not
  earning its capacity; inspect routing imbalance and only then consider a
  parameter-matched shared predictor.

The result must be interpreted jointly with the five-seed and per-rate table,
not from one seed or one missing rate.
