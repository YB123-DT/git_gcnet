# Implementation verification

No five-seed training launched. No accuracy improvement claimed.

## Files

- `gcnet_missing_m3/osram.py`: LocalCenteredContextFusion and opt-in final readout; legacy flat modules retained.
- `gcnet_missing_m3/model.py`: fusion configuration forwarding and incompatible-mode checks.
- `gcnet_missing_m3/train_gcnet.py`: CLI/config/provenance, independent per-rate checkpoint saving/restoration, rate-scoped diagnostics.
- `tests/test_osram_local_gated.py`: active masks, padding, gate sensitivity, zero initialization, two-step gradients, scan/context/projection identity.
- `tests/test_osram_per_rate_selection.py`: per-rate winners, earliest ties, saved/restored checkpoint identity, no global checkpoint, legacy selection compatibility, gated diagnostics scope.
- This experiment directory: approved plan, design, one-batch smoke code/output and pytest evidence.

## Tests

258 pytest cases passed in35.18seconds,0 failures. Existing PyG deprecation warning only. `pytest.xml` records the run. Suites include causal OSRAM, emotion-only masking, write-step, historical forward/backward/RNG/state compatibility, retention, write intervention, Missing-M3/JEPA and B2 compatibility, plus the two new suites. Historical source code was supplied through existing environment hooks because the remote code copy has no .git; no environment installation or repeated GPU smoke suite was needed.

Core tests were first observed failing because LocalCenteredContextFusion did not exist; then passed with implementation. Trainer tests likewise verified the absent new behavior before implementation. Read-only independent review found no blocking issues; flat-to-gated generic warm-start remains intentionally unsupported.

## One real MOSI batch

GPU2, FP32,32conversations,843valid utterances, natural miss=.5, causal eta=.6, current frozen features512/1024/1024. Only one batch was used, with two optimizer updates on that same batch; not a1epoch or dataset training run. Initial smoke invocation selected the fold list instead of its loader and failed before model forward; this fixture indexing was corrected before the successful run.

- Real existing Flat seed66 checkpoint loaded strict=True with old config defaulting flat.
- At fixed shared weights, all63memory write states are exactly equal across Flat and gated readout.
- Returned Local/Base/Gap and both structured predictor heads are exactly equal.
- Teacher was not called by classification/prediction forward; teacher parameters received no gradient.
- All backward gradients finite, original memory key-projector gradients nonzero.
- First update: gate/q/k/v emotion gradients0, context_out nonzero, as required by zero initialization.
- Second update: emotion-gradient L1 sums q22.5222, k32.7317, v181.7128, gate73.3100, context_out360.4364; all nonzero.
- Second forward active-only gate means Base0.49235, Gap-A0.48955, Gap-T0.47298, Gap-V0.50640. Mean active count2.35469; residual norm0.44637; subject norm72.27776; mean ratio0.006448.

These are initial smoke diagnostics, not learned evidence-use conclusions. The two total losses were1.6867 and8.7410: finite execution/gradient reachability is verified, but loss decrease, convergence, and downstream improvement are NOT established by this smoke. Raw measurements are in `SMOKE.json`.

`git diff --check` and syntax compilation passed before commit. Formal training remains stopped awaiting user confirmation.
