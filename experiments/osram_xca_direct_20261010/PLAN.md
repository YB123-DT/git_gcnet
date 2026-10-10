# XCA direct evidence transform followed by original Flat

INTERNAL DIAGNOSTIC ONLY. User approved XCA and then added Neural Production:
two independent seed66 MOSI100epoch runs, same protocol, healthyGPU7 concurrent.

Goal: test XCA when it must transform adapter evidence rather than provide a
parallel residual. Not a claim that residual suppresses useful learning.

Original XCA: typed five-role64d projections -> four-head XCA -> active mean64d
-> zero-init64-to1600 residual -> add original Flat anchor -> emotion_norm/head.
New: same typed projections and XCA channel update, retain five updated role
tokens rather than mean, normal-init local64-to256 and shared memory64-to512
decoders -> replaced adapter Local/Base/Gap -> original Flat Adapter.
Final: head(LN(local_skip(original L)+adapter([L',B',G']))).
No output evidence addition. Original Local Skip remains. First/empty-history
rows use original inputs (existing MeaningfulInputAdapter semantics).
Inactive Gap and padding safely masked; backward halves untouched (cfg84 zero).
The XCA channel softmax is unchanged; no Gate/completion/new loss or Memory change.
Slot decoding differs from original pooled residual, and parameter counts differ;
therefore not a parameter-matched removal-only ablation.

Files: priority40_pooling_geometry.py (expose existing role update, retain old
pool calculation); meaningful_input_priority40.py (new direct builder/class);
priority40_registry.py (input variant); core20/run.py (route config and verification);
tests/test_xca_direct.py (mask/direct/parity/gradient); local experiment dispatch.

- [ ] Write tests for registration and slot transforms, run RED.
- [ ] Expose masked role encoding; pool_roles remains active mean of it.
- [ ] Implement direct decoding with no input addition; retain shared Flat.
- [ ] Run CPU regression, masks/NaN/padding, finite gradients and core update.
- [ ] Commit/push, sealed tracked source on biggpu, healthyGPU7 admission.
- [ ] Start only one100epoch run; save config/source/data/GPU/PID/log provenance.
- [ ] Completion: verify100epochs, eight BEST/predictions, original mask hashes;
  compare per-rate/mean8/high to original Flat and original XCA. No auto multiseed.

Protocol: reuse same-seed original cfg84 Flat config, Adam1e-3 constant,
batch32, weight_decay1e-5, original taskMSE, emotion-only, cyclic0.0–0.7,
per-rate Test-oracle BEST. No local-LR/warmup/random Nested recipe bundled.
Reference: original Flat81.068095/76.352251; XCA81.041519/76.257088 W-F1%.

NPS addition: original four rules, two internal rule-update steps and real head
tokenizer unchanged. Keep original Local unchanged. Normal-init output bridges
directly replace Base/Gap, rather than original evidence+decoded output. Internal
production token update still uses its original additive rule mechanism; only
the external evidence bypass is removed. No auxiliary loss. Same parameter
count/RNG draw sequence as original NPS; output initialization is changed.
Reference NPS seed66 mean8/high80.981287/76.475237. Register
`neural_production_direct` and reuse TokenReadout via explicit residual=False,
zero_decoder=False. Add regression proving decoder zero gives zero Memory but
leaves Local unchanged, old residual default unchanged, finite gradients.
