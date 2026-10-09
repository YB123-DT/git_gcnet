# Flat256 + Nested704 near-equal parameter-budget test

INTERNAL DIAGNOSTIC ONLY

Status: RUNNING; verified first completed epoch and live training PID. User explicitly authorized implementation/training after DESIGN.md. Source snapshot `a3e4a9b`; start2026-10-09T10:49:09.094495 UTC; dispatcher PID1153498, training PID1153996; tmux `nested_equal_budget_20261009`. GPU3 free32495MiB at admission. Effective RAW_CONFIG verified: Flat256, nested_dim704, seed66/100epochs, original regression/MSE and emotion-only. Final scores pending.

One run only: MOSI seed66,100epochs, original cyclic random missing0–.7, emotion-only regression/MSE, Adam .001, weight decay1e-5, batch32, per-rate BEST Test-oracle matching the preceding capacity series. No freezing, new loss, completion, JEPA, mask or task-head changes. Baselines reused, not retrained.

Flat4352->256->1600; Nested704-state,704-hidden MLPs,3GIN layers, original rooted mean pooling/topology. OSRAM remains8x64; graph states are widened, not Memory values. Original zero-init decoders return Local256 and each Memory head64. Existing generic NestedSweepGNN implements this; no graph algorithm rewritten.

| Configuration | Total parameters | Mean8 W-F1 | High W-F1 |
|---|---:|---:|---:|
| Original large Flat |13509793|81.068095|76.352251|
| Flat256 only |5508961|79.529424|74.421931|
| Flat256+Nested64 |5668196|80.362122|75.899420|
| Flat256+Nested704 |13560676|pending|pending|

New Nested module8051715 parameters; total measured by constructing the actual model. +50883 (+0.377%) versus original large Flat, not exact equality. Equal parameter budget is not equal FLOPs or activation memory. Graph-internal initialization differs with dimensions; common backbone/Flat state and outer seed scheme retained.

## Verification

- Red test: unknown `small_flat_nested_dim704` rejected before registration.
- CPU pass: full/module counts; exact common-parent initialization; zero-init output equality; three finite optimizer steps; GIN weights updated; inactive Gap NaN injection does not affect output; inactive outputs and padding zero; first-turn Local/Base pass through.
- Real batch CUDA pass on healthyGPU3: largest valid-utterance batch among seed66 initial training loader batches, original task at cyclic epoch7, then .7 evaluation. Peak allocated8658.397MiB/reserved9596MiB. This is a sampled resource check, not a guarantee of worst-case100epoch peak. No scores used to choose architecture.
- Only this method's launcher admission raised from8048 to12000MiB free; all previous launch behavior preserved. Formal run uses healthyGPU3 with ample free memory and no new competing jobs. No batch or precision changes.
- Python compile and diff whitespace checks passed. Unrelated `gcnet/model.py` edits not included.

## Launch command and artifacts

Server biggpu, physicalGPU3 UUID GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a; GPU4 prohibited.

```sh
python -m experiments.osram_nps_local_20261009.dispatch \
  --method small_flat_nested_dim704 \
  --root /data2/yb/remote_experiments/osram_nested_equal_budget_20261009 \
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json \
  --gpu-index 3 --gpu-uuid GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a
```

Run in immutable git archive sealed with SNAPSHOT.json. Independent seed_66 logs/history/config/provenance, eight BEST checkpoints/predictions and last_training.pt. Verify exit code and artifact hashes on completion. No automatic capacity/seeds expansion even if results are poor.
