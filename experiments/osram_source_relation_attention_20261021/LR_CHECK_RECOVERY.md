# MOSI epoch101 LR-audit roundoff recovery

INTERNAL TEST-ORACLE SCREEN. This is an engineering recovery, not a new LR trial.

The original immutable6dbcc98 worker completed/persisted100epochs, then raised
`Reference training learning_rates changed` at101 because exact list equality
compared3e-5 with the Flat record's3e-4/10=2.9999999999999997e-5.
Difference3.3881317890172014e-21. Source/sampler/data/mask/update counts agree;
this is not a task failure or a changed intended schedule. Epoch101's uncommitted
optimizer updates are discarded by restoring the full epoch100 state.

Fix only the audit comparison: LR floats use math.isclose(rel_tol1e-14,abs_tol0),
all epoch/mask/update checks remain exact. The actual scheduler, training setup,
loss, model/source tracker/attention, optimizer and clipping functions are
unchanged. AST checks confirm settings/set_learning_rates/setup/original_digest/
validate_gradients/compact_diagnostics identical. All non-worker source hashes
must match exactly before explicit migration is allowed.

The source migration is narrowly pinned to CMUMOSI seed66,epoch100,original
source6dbcc986c451688ea33f33d3ffe6785e0be775f5 and worker SHA256
64b17f0c5cca283b2b9f2c737457e48f73bd28833556f1b2f515b22f5536ce35.
It rejects config,mask,split,data,seed or other-source changes. Normal resume
still requires exact identity; no general strict-check bypass was introduced.

Before continuation, preserve the entire original MOSI run (including eight
BEST+last/RNG/history/failure metadata) to the separate `recovery_epoch100`
directory. Resume the same run output from last.pt, never BEST, in a new
immutable source snapshot; verify model bytes/RNG exactly after restoration.
The original MOSEI process and source snapshot are untouched.

RECOVERY.json and train_CMUMOSI_resume.log track the new worker. The original
QUEUE.json retains its original failed-process return code for audit; it is not
the authoritative recovery status. The recovery watcher waits for both resumed
MOSI and original MOSEI and regenerates the final report after the original
controller exits. Mixed source versions differ only in audit logic; full source
lineage remains in each checkpoint and recovery record, not relabeled as one SHA.

Five focused runner tests passed, including the reproduced roundoff boundary,
rejection of real LR/mask changes, rejection of incompatible source/config/epoch
migrations. No fresh training run, smoke rerun, mask resampling or HPO is added.

The first recovery bootstrap was safely rejected before any optimizer step:
the reconstructed data manifest's origin path moved from source_6dbcc98 to the
new immutable snapshot, despite identical data/manifest fingerprints. The second
recovery admits only that path relocation after hashing both full manifest files
and comparing all other manifest metadata. It still rejects changed content,
feature-file hashes or data split. RECOVERY.json retains the failed bootstrap;
RECOVERY_v2.json tracks the retry. No weights/masks were changed by the rejection.

Recovery verified: new immutable89014b4 snapshot matches9592trackedGitblobs.
All6focused tests pass. Full epoch100 model/Adam/RNG restored exactly, manifest
content hashes checked, all non-worker source fingerprints unchanged. MOSI
passed and persisted epochs101–104 with no audit error. MOSEI's original PID
2565343 remains unchanged, observed15/60epochs. See RECOVERY_VERIFIED.json;
both final performance results remain pending.
