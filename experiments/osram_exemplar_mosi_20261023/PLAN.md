# MOSI TRAIN-exemplar transfer implementation plan

Approved scope: MOSI seed66, feature-only versus TRAIN-label-aware retrieval;
two new150-epoch jobs, reuse same-protocol Flat66. No other datasets/HPO/seeds.
Design source: `../osram_exemplar_transfer_review_20261023/DESIGN.md`.

Architecture: keep original CHM and Flat; expose an optional pre-emotion_norm
callback with no parameters or default-path arithmetic changes. A separate
wrapper owns the retrieval module and a TRAIN-only raw feature bank. Shared
ObservedSetEncoder/Local encode supports with query availability; exclude the
entire query conversation. Current TRAIN labels are not wrapper inputs.

Files and ownership:
- `gcnet_missing_m3/exemplar.py`: retrieval arithmetic, support bank, wrapper.
- `gcnet_missing_m3/model.py`, `osram.py`: optional callback plumbing only.
- `tests/test_exemplar.py`: red-first retrieval/masking/gradient/default tests.
- `experiments/osram_exemplar_mosi_20261023/{worker,run,report}.py`: runner,
  fixed baseline masks, provenance, finite persistent queue and reporting.
- `tests/test_exemplar_runner.py`: finite budget/config/selection tests.

## Execution

- [ ] Write tests before implementation: use tiny deterministic tensors to
  assert same-conversation exclusion, zero-neighbor result, no inactive feature
  influence, feature-only label invariance, label-aware gradient and RNG safety.
  Run `python -m pytest tests/test_exemplar.py -q`; observe missing feature failure.
- [ ] Implement `TrainExemplarReadout` with shared LayerNorm/Key, exact top16
  negative squared-L2, `Linear(y)+T(k-k_i)`, and128→1600 output. Keep TRAIN-only
  bank manifest independent of checkpoint parameters. No new dependencies.
- [ ] Add callback path in original Flat-only branch; test original disabled
  state_dict/output/RNG identity, one scan, eval target-label independence,
  finite gradients and model/bank save-reload identity.
- [ ] Runner reuses accepted setup, sampler and persist_epoch; Adam/batch32/
  WD1e-5/MSE; LR3e-4 for100epochs then3e-5 for50. Retain eight exact BESTs,
  same-call masks/predictions plus full last model/optimizer/RNG/bank identity.
- [ ] Run unit/integration checks in the original s0 environment; then real
  MOSI GPU smoke for both arms. Check dataset SHA, baseline initial digest,
  masks and optimizer updates against accepted reference. Record actual cost.
- [ ] Commit implementation with Lore trailers; archive immutable tracked
  source on biggpu. Run only two jobs with UUID binding on a healthy GPU,
  resource-gated concurrency; never use physical GPU4 or alter other jobs.
- [ ] Summarize complete/partial status truthfully and push small records;
  scores are INTERNAL TEST-ORACLE SCREEN, not independent test evidence.

Stops: failed invariance or smoke prevents training. Numerical/operational
failures require recorded diagnosis; no silent protocol change or automatic HPO.
