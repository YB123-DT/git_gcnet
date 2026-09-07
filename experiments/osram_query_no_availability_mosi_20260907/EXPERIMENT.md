# OSRAM Query `a_t` A/B diagnostic

This is an internal backbone ablation, not a formal paper result.

- Dataset: CMU-MOSI, fold 1
- Seeds: 66, 67, 68, 69, 70
- Missing rates: 0.0–0.7, cyclic mixed-rate training
- Features: `wav2vec-large-c-UTT`, `deberta-large-4-UTT`, `manet_UTT`
- Backbone: OSRAM H8 / key 32 / value 32 / output 700
- Objective: current joint classification + JEPA objective
- All optimizer, Student/Teacher, MMoE, masks, batch size and learning-rate
  settings are unchanged.
- Selection: per-rate Test-oracle extraction for diagnosis; strict one
  eight-rate-mean Test-oracle checkpoint per seed is also reported as a guard.

The explicit-`a_t` reference is inherited from
`experiments/osram_heads8_out700_20260906/`. Only the no-explicit-`a_t`
condition is newly trained with `--no-osram-query-availability`.

Remote output:
`/data2/yb/remote_experiments/osram_query_no_availability_mosi_20260907`
