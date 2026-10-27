# MOSEI66 frozen Nested: three single-factor structures

Finite user-approved follow-up to the completed two-LR screen. Only24 new jobs:
three structures ×eight own-rate frozen parent/branch pairs. No new Flat, seed,
dataset, LR search, loss, graph topology or training-missing changes.

| Config | Node dim | Depth | Internal MLP | Groups | Actual trainable params |
|---|---:|---:|---:|---:|---:|
| Protected default, reused |64|3|64|8|159235|
| dim32 |32|3|64|8|80163|
| mlp128 |64|3|128|8|216899|
| depth2 |64|2|64|8|146690|

IMPORTANT: dim32 uses `mosei_nested_d32_l3_g8_m64`. The plain registered dim32
alias would implicitly shrink the MLP to32 as well, violating this single-factor
design. All three methods already exist in the pinned original numerical source;
they are capacity comparisons, not new mechanisms. No core file is modified.

`VariantFrozenNestedModel` reuses the original frozen wrapper's forward/decode,
and constructs only its chosen adapter (no discarded default adapter RNG draws).
The dimension64/depth3/MLP64 sweep implementation matches the original Nested's
seeded state tensors and nonzero-decoder output exactly in the CPU parity test.
Changed-shaped variants are not claimed to share the default parameter tensors;
their own seed66 initialization is reproducible and zero decoders reproduce Flat.

Fixed training: cmerc-repro, original MOSEI66 own-rate Flat checkpoints trained
at1e-4, all parent parameters frozen/eval. New Nested only: constant3e-4,60epochs,
Adam,batch32,WD1e-5, original clipping/MSE including neutral labels, cyclic
gcnet-random-mask-v1. Only own-rate original exact selected test mask is evaluated,
W-F1 strict improvement/earliest tie. Original LocalSkip remains unmodified.
CHM uses original8×64heads, jointATVwrite,read-before-write,write_step0.6.

Every formal epoch must preserve the protected default Nested's sample/mask/
optimizer-update trace. Selected checkpoint/predictions are hash-checked and
strictly replayed; parent state/gradient, actual architecture, parameter count,
source fingerprints and all-group learning rate are checked. Full recovery
requires block/Adam/RNG/history/config identity; no weights-only fake resume.
Original Signature/NeuMiss weights remain deleted and are not dependencies.

Server ssh biggpu. Python `/data2/yb/conda_envs/cmerc-repro/bin/python`.
Numeric cwd `/data2/yb/remote_experiments/osram_signature_cyclic_multi_20261025/source_e0ca94c`.
Independent immutable overlay contains this package plus reused LR screen,
frozen_nested_gcnet, frozen_nested_cross and recent_noarea_serial helpers; no
editable gcnet source is overlaid. Old experiments and checkpoints remain intact.

```sh
cd NUMERIC_CWD
PYTHONPATH=IMMUTABLE_OVERLAY:$PWD \
 /data2/yb/conda_envs/cmerc-repro/bin/python -u \
 -m experiments.osram_mosei_frozen_nested_capacity_20261027.scheduler \
 --root NEW_RUN_ROOT --source-commit ACTUAL_FULL_COMMIT --per-gpu 4
```

tmux-persistent controller first executes three short GPU smokes (two single-batch
updates each), then24 formal jobs. Healthy physicalGPU0/1/2, true UUID binding,
maximum4/card and12global. Actual free VRAM/RAM/disk plus conservative reserves
gate admission. No changes to batch to fit; no use of faultyGPU4 or other tasks.
Failures stop new admissions and preserve live workers; no hidden retries or
hyperparameter substitutions. QUEUE.json stores PID,command,cwd,UUID,logs/status.

After all24finish, the collector automatically validates actual variant parameter
counts (not the original159235), selected-mask alignment, parent predictions,
same sampling traces and recomputed metrics. RESULT.md/JSON/CSV compare each
variant with protected Flat and defaultNested on eight-rate and high-missing
ACC/W-F1, plus every individual rate/epoch/correction/harm and cost.

INTERNAL per-rate Test-oracle, seed66only. Eight-rate averages combine separate
own-rate parent/branch pairs; they are not one deployment checkpoint. Extra
stage2 training budget and single-seed selection remain limitations. No automatic
new seeds, combined structures, other rates or additional HPO after this scope.
