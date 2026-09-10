# Local-Centered Gated Context Integration

Implementation and one-batch verification only. **Five-seed formal training has NOT been launched.**

## Bounded architecture

Only the mapping from existing Local/Base/Gap to emotion hidden changes. Local remains `read_node + local_path(read_node)`. The original scan, read-before-write, queries, keys, values, alpha/beta, ridges, eta=.6, Base/Gap tensors, MMoE/JEPA and classifier are unchanged. B2 and older ablation code remain available but cannot be combined with this first gated variant.

`osram_readout_fusion` is `flat` (default) or `local-gated`. `flat` does not instantiate new parameters, consume additional initialization RNG, or change historical state keys/numerical forward. Old configs without the field select flat. Gated construction isolates its RNG so shared modules initialized afterward do not shift. Old flat emotion adapter/skip/norm remain registered for historical code, but are frozen/unused under gated mode. The new subject projection and output norm start as copies of their original counterparts.

## Equations

Evidence order: Base, Gap-A, Gap-T, Gap-V. Base is active on valid utterances; Gap-m is active only on valid utterances missing modality m. Inactive tensors are explicitly cleared before context LayerNorm, and their gates are explicitly zeroed afterward. This handles nonzero or even nonfinite inactive inputs without contaminating output.

```
q = Linear(LN(local))
k_i = shared_key(LN(context_i))
v_i = shared_value(LN(context_i))
g_i = active_i * sigmoid(MLP([q, k_i, q*k_i, abs(q-k_i), type_i]))
delta = sum_i(g_i*v_i) / max(sum_i(active_i), 1)
hidden = LN(local_skip(local) + context_out(delta))
```

Interaction dimension128, type embedding4×16. Gate MLP: Linear(528,128), GELU, Linear(128,1). No softmax, competition normalization, multihead attention or uncertainty interpretation. Value/key projections are shared across evidence types. Context output: Dropout(original dropout), Linear(128, output_dim), with final weight and bias zero-initialized. Local never enters a large joint concat-reencoding branch.

The residual formulation preserves an explicit Local subject path; it does not mathematically guarantee context will never dominate later in training. Residual/subject norm is monitored rather than constrained by a new loss.

## Shapes and parameters

Current MOSI: Local [L,B,256], Base [L,B,512], Gap [L,B,3,512], availability [L,B,3], umask [B,L], output [L,B,700]. Forward-only retains existing zero backward slots unchanged.

| Count | Flat | Local-gated |
|---|---:|---:|
| Registered total, including frozen modules/teacher | 7,181,217 | 7,686,482 |
| Trainable | 6,321,057 | 4,536,214 |
| New fusion parameters | — | 505,265 |

The new module adds505,265 registered parameters, including copied subject projection/norm. It replaces2,290,108 active flat-readout parameters, which remain stored but frozen; trainable parameters decrease by1,784,843. This is not a parameter-matched experiment, and an improvement alone would not distinguish gating from reduced trainable capacity. No parameter-matched control is added in this round.

## Diagnostics

Active-only mean gates for Base/Gap-A/Gap-T/Gap-V; absent evidence groups are `None`, never fake zero means. Also: valid-utterance mean active evidence count; mean context residual norm; mean Local subject norm; mean per-utterance residual/subject norm ratio. Stored tensors/statistics are detached and never enter loss or control.

Per-rate selected-checkpoint diagnostics are saved separately. Current generic diagnostic snapshots are explicitly **last evaluation batch**, not a dataset-wide mean. This is sufficient for the current MOSI test set fitting one batch32, but must not be mislabeled on larger datasets.

## Selection and future experiment (not launched)

Use `--osram-readout-fusion local-gated --checkpoint-selection test-oracle-per-rate`. For gated experiments the trainer rejects the old mean-selection mode. Each seed/rate independently selects maximum Test weighted-F1, keeping earliest epoch on ties, and saves `best_miss_0p0.pt` through `best_miss_0p7.pt`. Final per-rate predictions restore the corresponding checkpoint. No global best.pt or global selected epoch is produced in this mode. Arithmetic means in logs/descriptive summaries never drive selection.

Checkpoint metadata contains selection_split=test, selection_protocol=per-rate-test-oracle, selection_rate, selection_weighted_f1, epoch and full config. metrics.json contains selected_epoch_by_rate and selected_weighted_f1_by_rate. Legacy validation and mean-oracle modes remain unchanged for historical experiments.

Planned comparison after approval: inherited Flat vs fresh Local-gated, MOSI seeds66–70,100epochs,cyclic,forward-only,eta=.6, all other settings from the corresponding Flat config. Per-rate means±sample SD, paired deltas, positive seeds/5; all-eight and high-missing means are descriptive only. **INTERNAL TEST-ORACLE DIAGNOSTIC ONLY; NOT A FORMAL PAPER RESULT.** If flat is matched, no evidence supports added readout complexity; if gated loses, stop fusion complexification rather than adding attention.

## Compatibility scope

Old flat checkpoints load strict=True and retain exact paths. New gated checkpoints retain their fusion field and own new state keys. Automatic warm-starting gated from a flat `initial_backbone_checkpoint` is not part of the planned from-scratch experiment; that generic loader deliberately rejects missing fusion keys. The one-batch smoke uses an explicit checked shared-key transfer for verification only, not a proposed training protocol.
