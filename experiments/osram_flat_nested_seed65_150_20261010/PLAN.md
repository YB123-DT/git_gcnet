# Seed65 Flat versus Nested, 150 epochs

INTERNAL DIAGNOSTIC ONLY. User requests seed65 result after150epoch comparison.
No seed65 Flat/Nested run found in local tracked experiment records or remote
/data1,/data2 remote_experiments (directory search depth6).

Use same two-stage LR budget as the preceding comparison: original constantLR
1e-3 epochs1–100 from scratch, then FULL-state continuation at1e-4 epochs101–150.
These are not150epochs at constant1e-3. One seed, two models only; no architecture
change or other new experiments. cfg84 no-JEPA MOSI cyclic random missing,
task MSE/Adam/batch32/clipping1/per-rate Test-oracle selection unchanged.
Use sealed source_ad211c0 and data manifest previously checksum-verified.
Flat seed66 monitored100 config supplies the reference; onlyseed and Nested
switch change. Reuse passive observer (no extra backward/RNG) during first100.

Run biggpu physicalGPU7, UUID GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e,
two parallel persistent pipelines. GPU4 forbidden. Independent /data1 directories
for original100 and continued150; source checkpoints never overwritten.
Prelaunch GPU7 free32495MiB; /data1 free85GiB. Preserve full recovery and8 BEST.

Runner defaults seed66 remain unchanged. New --seed65 restricts override to seed;
skip comparison to seed66 evaluation-mask hashes because seed defines masks.
Final Flat65/Nested65 must have identical evaluation masks and passive per-step
training availability hashes for a paired comparison; compare actual100 and150
budgets and new-only101–150 separately. Do not reuse seed66 scores as seed65.

Persistent launch runs original100 wrapper, then existing low-LR continuation
only if100 completes successfully. Failure prevents second stage. Preserve logs,
runtime configuration/provenance and exact commands. Results pending.
