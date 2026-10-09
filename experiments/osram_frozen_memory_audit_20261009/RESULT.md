# Frozen Memory information and history intervention audit

INTERNAL DIAGNOSTIC ONLY

Status: implementation and limited real-checkpoint verification complete; full eight-rate diagnostics preparing. This is not an OSRAM retraining run.

User override: no validation fitting, evaluation or selection. Train-only probe gradients, Test-MSE-selected probe checkpoints. Both backbone and probes have used test selection. Results cannot establish independent generalization, natural-language causation, or the semantic identity of a latent direction.

## Models and scope

- Original cfg84 no-JEPA LARGE Flat seed66; same eight existing per-rate BEST checkpoints under `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66`.
- No Local/Memory/Query/task-head modifications, no new main-model losses or missing-feature predictions.
- Cache real Local256 and active Base/Gap forward512; ignore permanently zero backward halves.
- A: Local+availability,33,409 trained parameters. B: Local+real Memory+availability,33,089. C: Local+train-donor Memory+availability,33,089. Fixed shared random512->64 memory projection; widths128 vs64 approximate parameter matching, not identical architectures. Weak probe results do not establish information absence.
- Probe seeds66/67/68 are head initializations, NOT three independently trained backbones. Four target tasks: current score, previous score, past-three mean, current-minus-previous score. Only current task's sign metrics represent current sentiment classification.
- Historical gold labels are offline targets only. Same/other speaker comparison requires genuinely multi-speaker conversations; MOSI single-speaker cases are unavailable, not negative evidence.

## Verification

- Unit tests cover cache shape/identity/finiteness/first-turn and inactive Gap safety, donor constraints, causal targets, test-label gradient isolation, frozen paired scans, deterministic one-bit interventions, strict past-only masks and count matching.
- Real checkpoint check at rate .7 passed:686 utterances/656 nonneutral; ACC75.914634%, W-F1 **75.773178%**, identical original metric. Original checkpoint SHA256 `1cc1c155c57c36879db60c6b63b668951ed679de6809e96e45bd2117d48dce68`; frozen model hash unchanged.
- Small check:12 miniature head fits (2epochs, one probe seed),9 paired interventions; maximum same-current Local absolute difference `9.5367431640625e-07`, within1e-5 numerical tolerance. These miniature heads are NOT performance results. Check artifacts: `/data2/yb/remote_experiments/frozen-memory-check-GvpYlz/check_output`.

## Full run

Two disjoint rate shards, GPUs2/3 on biggpu (GPU4 excluded), healthy UUID explicitly pinned. One fixed protocol,100epochs per head, no sweep. Immutable source and independent logs/output paths. Each unique past deletion is scanned once and reused for all eligible target utterances, without sharing state between different histories.

```sh
python -m experiments.osram_frozen_memory_audit_20261009.launch \
  --root /data2/yb/remote_experiments/osram_frozen_memory_audit_20261009
```

After both shards finish, the launcher runs `analyze` automatically and writes `summary/RESULT.md`, `SUMMARY.json`, per-rate/task/seed probe CSVs, matched corrections/harms and historical intervention tables. Separate exact-lag modality contrasts from approximate lag matches. Preserve every unavailable/failed case and distinguish partial from complete summaries. No conclusion yet.
