# Frozen Memory information and history intervention audit

INTERNAL DIAGNOSTIC ONLY

Status: COMPLETED on biggpu at2026-10-09 12:22:20UTC; started12:13:52UTC. Both shards and analysis exited0. Eight rates,288 offline probe fits and22,419 target/intervention-family/rate pairs completed (not22,419 unique utterances). All640 recorded output artifact hashes independently rechecked before GitHub delivery. This is not an OSRAM retraining run. Source snapshot `f2f5d0f`; exact commands/logs/UUIDs preserved in LAUNCH.json.

## Completed findings: three diagnostics

Detailed machine-readable output and CSV tables: [summary](summary/RESULT.md), [SUMMARY.json](summary/SUMMARY.json). All below are descriptive Test-oracle results, not independent-test claims.

### 1. Additional current-sentiment predictive signal

W-F1 percentages, averaged over rates then three probe initializations:

| Probe | Eight-rate W-F1 | High-missing W-F1 | Eight-rate MSE |
|---|---:|---:|---:|
| A: Local + mask |78.082|72.511|1.453662|
| B: Local + real Memory + mask |79.785|74.572|1.371234|
| C: Local + train-donor Memory + mask |77.852|72.239|1.493912|

B exceeds A by1.703pp overall and2.061pp at high missing. It also exceeds the donor control. This supports additional readily decodable predictive signal under the specified probe design, not information-theoretic uniqueness or improvement over the original task head.

### 2. Fixed-current, paired historical deletion

Deleting the nearest eligible historical Text observation vs the next-earlier eligible Text observation increases current MSE by an eight-rate macro mean of0.036188 vs -0.001813 relative to original history. The nearest-Text intervention has120 corrections and161 harms across4,289 eligible target/rate pairs. These are repeated evaluation pairs, not unique independent examples. Not every deletion is harmful.

Exact-lag modality comparisons are also retained: T-vs-A deletion MSE increases0.002097 vs0.000026; T-vs-V increases0.014282 vs0.001205. These two comparisons have different eligible cohorts and must not be subtracted from each other. Per-rate conversation-bootstrap intervals and matched counts are in `summary/interventions.csv`; the macro averages alone are not significance tests. These are model-behavior interventions, not natural-language causal effects.

### 3. Historical information decodability

Eight-rate MSE (lower is better):

| Offline target | A: Local | B: Local+Memory | C: donor control |
|---|---:|---:|---:|
| Immediately previous score |2.454131|1.463803|2.503034|
| Preceding up-to-three score mean |1.558182|0.760718|1.605239|
| Current-minus-previous score |2.745819|2.287243|2.815810|

Gold history labels are offline targets only. These results show improved decodability with Memory; they do not identify which latent coordinates cause task improvement or establish emotion-shift detection. The same/other-speaker comparison is unavailable in the single-speaker MOSI conversations, with explicit coverage reasons preserved.

Together, historical readouts contain decodable sentiment-related signal, and changing past observations can affect current predictions. The experiment does not yet establish that the particular decoded property is the mediator of the prediction change.

User override: no validation fitting, evaluation or selection. Train-only probe gradients, Test-MSE-selected probe checkpoints. Both backbone and probes have used test selection. Results cannot establish independent generalization, natural-language causation, or the semantic identity of a latent direction.

## Models and scope

- Original cfg84 no-JEPA LARGE Flat seed66; same eight existing per-rate BEST checkpoints under `/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66`.
- No Local/Memory/Query/task-head modifications, no new main-model losses or missing-feature predictions.
- Cache real Local256 and active Base/Gap forward512; ignore permanently zero backward halves.
- A: Local+availability,33,409 trained parameters. B: Local+real Memory+availability,33,089. C: Local+train-donor Memory+availability,33,089. Fixed shared random512->64 memory projection; widths128 vs64 approximate parameter matching, not identical architectures. Weak probe results do not establish information absence.
- Probe seeds66/67/68 are head initializations, NOT three independently trained backbones. Four target tasks: current score, previous score, past-three mean, current-minus-previous score. Only current task's sign metrics represent current sentiment classification.
- Historical gold labels are offline targets only. Same/other speaker comparison requires genuinely multi-speaker conversations; MOSI single-speaker cases are unavailable, not negative evidence.

## Verification

- 21 unit tests plus4subtests passed, covering cache shape/identity/finiteness/first-turn and inactive Gap safety, donor constraints, causal targets, test-label gradient isolation, frozen paired scans, deterministic one-bit interventions, strict past-only masks and count matching.
- Real checkpoint check at rate .7 passed:686 utterances/656 nonneutral; ACC75.914634%, W-F1 **75.773178%**, identical original metric. Original checkpoint SHA256 `1cc1c155c57c36879db60c6b63b668951ed679de6809e96e45bd2117d48dce68`; frozen model hash unchanged.
- Small check:12 miniature head fits (2epochs, one probe seed),9 paired interventions; maximum same-current Local absolute difference `9.5367431640625e-07`, within1e-5 numerical tolerance. These miniature heads are NOT performance results. Check artifacts: `/data2/yb/remote_experiments/frozen-memory-check-GvpYlz/check_output`.
- Full completion verified all eight rates: frozen state unchanged and original baseline metric parity passed throughout. Probe gradients use training rows only; validation was not evaluated or used for selection.

## Full run

Two disjoint rate shards, GPUs2/3 on biggpu (GPU4 excluded), healthy UUID explicitly pinned. One fixed protocol,100epochs per head, no sweep. Immutable source and independent logs/output paths. Each unique past deletion is scanned once and reused for all eligible target utterances, without sharing state between different histories.

```sh
python -m experiments.osram_frozen_memory_audit_20261009.launch \
  --root /data2/yb/remote_experiments/osram_frozen_memory_audit_20261009
```

The launcher automatically completed `analyze`, producing `summary/RESULT.md`, `SUMMARY.json`, per-rate/task/seed probe CSVs, matched corrections/harms and historical intervention tables. Exact-lag modality contrasts are separate from approximate lag matches. Summary reports `partial=false` and `verification_passed=true`. Raw features, model weights and per-utterance caches remain on biggpu; only code, aggregate reports and launch provenance are uploaded to GitHub.
