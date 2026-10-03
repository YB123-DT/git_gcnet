# Authorized three-seed extension

INTERNAL DIAGNOSTIC ONLY

The user requested multi-seed checking after the seed66 negative result. Reuse
completed seed66; add seeds67/68 only, each from scratch for100 epochs. Keep the
same 0.5 base + 0.5 full objective and original cfg84 protocol. No hyperparameter
search, new modules or repeated seed66 training.

Status: seeds67/68 launched on biggpu host GPU5 in parallel. Results pending.

|Seed|PID|Code commit|
|---|---:|---|
|67|733120|14bdda8|
|68|757004|14bdda8|

GPU UUID: GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62.
Remote root: /data2/yb/remote_experiments/osram_relation_dual_readout_multiseed_20261003.
An isolated code/ snapshot is used; seed_67/ and seed_68/ hold separate outputs.
Commands and PIDs are recorded in launch_seed_67.json / launch_seed_68.json;
logs are train_seed_67.log / train_seed_68.log. Training uses the previously recorded
command with --seed67 or --seed68 (separate CLI arguments), --dual-readout and
--commit14bdda8. The runner reads the corresponding original Flat seed config,
validates it, and checks the eight evaluation mask hashes on completion.

Launcher verification: all three supported seed configurations preserve baseline
fields; mismatched requested/reference seeds are rejected. No model code changed.
Live remote processes confirmed for both runs; this is not completion evidence.

After completion compare each seed with its paired Flat baseline and report all
three seeds, per-rate and8-rate/high-missing mean. Per-rate BEST remains Test-oracle;
do not present these internal comparisons as validation-selected paper results.
