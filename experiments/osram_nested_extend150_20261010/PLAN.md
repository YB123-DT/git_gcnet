# Extend original Nested to 150 epochs

INTERNAL DIAGNOSTIC ONLY. Authorized: original MOSI Nested only, seeds66/67/68.
No Flat/Gate training, model changes or new losses. Additional budget is 50 epochs
per seed, not 150 fresh epochs. Keep original100epoch artifacts untouched.

Use historical source_ad211c0 snapshot on biggpu and full epoch100 last_training
state: model, optimizer, RNG, cyclic schedule progress, history and eight BEST
references. Constant LR stays .001; no scheduler/scaler was used. Explicitly
rebind copied recovery identity for the sole epoch-budget change, retaining the
original identity and source/checkpoint hashes. Strict restore verifies actual
loader/mask identities and original model state keys.

Output: /data1/yb/remote_experiments/osram_nested_extend150_20261010/runs/seed_SEED.
GPU7 UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e, three concurrent runs.
Use independently copied recovery versions, not hard links. Do not overwrite the
original experiment or its provenance. Source is hash-checked against historical
provenance; wrapper is frozen separately. Original dataset/manifest unchanged.

Compare eight-rate/high-missing W-F1, per-rate scores and selected epochs at100
versus150. BEST includes earlier epochs: it cannot decrease simply by extending
the selection window. Any gain is therefore an internal extra-budget screening
result, not evidence of improved architecture or a fair equal-budget comparison.

Verification: source/data/checkpoint integrity, full100epoch state, exact sole
config delta, live epoch101+ training, unchanged masks and original checkpoint,
final150epoch history and all20 selected/recovery artifacts. No new GPU smoke.
