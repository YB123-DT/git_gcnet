# Independent meaningful-block integration review

Date: 2026-10-03. Reviewer: decision_impl, not the integration author.
Verdict: **APPROVE, subject to the separately required per-candidate CUDA
admission checks.** The float64 grouping compatibility issue was corrected by
its owner and independently reverified below. No unresolved production blocker
was found in this review.

## Scope

Reviewed `gcnet_missing_m3/{meaningful_blocks.py,model.py,osram.py,train_gcnet.py}`,
the canonical fixed-modality `_build_model` keyword addition,
`tests/test_meaningful_block_integration.py`, wrapper tests, and the
`training_resume.py` contract invoked by the new trainer hooks. Compared the
integration's source with the provided frozen pre-change `95e7dad` snapshot at
`/data2/yb/paper/osram-baseline-check.BJxzcu`. No Git commands, production edits,
remote actions, or formal training launches were performed.

## Static contract checks

- The only enabled readout change is original Local skip plus original Flat
  adapter, then the new zero bridge, then original emotion normalization and
  task head. All original parameters remain trainable. No scan, query, key,
  value, write, memory-cache, completion, or task-loss implementation changed.
- The new wrapper removes padding and the first valid utterance by valid-prefix
  count. It slices the actual forward `8 * 64` dimensions, excludes observed
  Gap roles, and hard-masks before entering a family. Learned bridge bias cannot
  reactivate excluded rows. Availability is validated only on valid rows.
- Branch construction is inside CPU Torch `fork_rng`; default `none` does not
  instantiate a branch. Existing `osram.*` optimizer grouping includes every
  branch parameter without duplication. No new optimizer, loss, or clipping
  rule is introduced.
- TrainConfig, direct model, and OSRAM constructors reject unknown methods,
  non-cfg84 heads, old candidates, non-flat/noncausal paths, forward-slot reuse,
  and incompatible readout/query/gate/decision adaptations. TrainConfig/model
  also enforce the relevant single-view, emotion-only, no-completion contracts.
  CLI, saved configuration, model construction, summaries, and canonical
  checkpoint reconstruction carry the independent new flag.
- Recovery binds the real sampler identity/indices/seed and mask configuration
  hashes after setup. Restore is before the next epoch's LR update, sampler
  epoch assignment, and stochastic data/model action. The fixed protocol uses
  no mutable scheduler/scaler object; its LR update is epoch-indexed.
- Selection, history, and all improving per-rate BEST saves occur before the
  epoch commit and existing per-rate `continue`. A completed resume skips
  training and uses committed selected checkpoints for final evaluation.
  Immutable model/selection references preserve earlier committed BEST values;
  the full state restores canonical BEST and effective history before RNG.

## Independent historical regression probe

Loaded both frozen `osram.py` and `model.py` as isolated temporary in-memory
modules, injected the historical OSRAM class into the historical model, and
used the canonical builder with only its newly added keyword removed for the
old constructor. This is not a comparison between two current implementations.

At genuine Local256 / eight64-dimensional-head / output1600 dimensions with
small synthetic raw input dimensions `(3,4,5)`, verified:

- Exact state-dict key set and all initial parameter/buffer values.
- Exact initialization Torch, Python, and NumPy RNG states.
- Strict historical-to-current `none` checkpoint loading.
- Exact train and eval predictions and subsequent global RNG states.
- Exact one-step Adam update of the historical and current `none` paths using
  a deterministic float64 control (no clipping in this parity-only fixture).

The enabled Perceiver branch additionally passed float64 exact initial
predictions, one scan/one original adapter invocation, raw shared-gradient
parity at `atol=1e-10, rtol=1e-8`, complete optimizer membership, and strict
rebuild/loading through the canonical builder.

## FP32 enabled-path evidence

Independently checked Slot Attention, NODE, and Equilibrium Aggregation as the
private-RNG, data-aware-initialization, and higher-order inner-optimization
cases. Each passed:

- One causal scan and one original Flat adapter invocation.
- Exact initial full-model predictions.
- Exact raw shared-parameter gradients in the probe (maximum difference zero;
  acceptance tolerance was `atol=1e-6, rtol=1e-5`).
- Complete, unique optimizer membership of every trainable parameter.
- Three finite synthetic task-loss Adam steps with original clipping, changing
  parameters in the bridge, core, original adapter, and observed encoder.
- Strict checkpoint reconstruction/loading.

These are correctness controls, not performance experiments. Enabled shared
optimizer updates need not equal Flat updates after global clipping because
the new bridge contributes to the clipping norm; the original rule is retained.

## Trainer-hook recovery control

Executed the real `run_experiment` and real `TrainingState` with a tiny mocked
model and synthetic train/evaluation functions, but the real epoch-seeded
sampler and conversation-mask schedules. The evaluation stub deliberately
consumed Torch/Python/NumPy randomness to stress the commit boundary. Interrupted
immediately after the first committed epoch, rebuilt, and resumed under the
same two-epoch configuration.

Compared with uninterrupted two-epoch execution, the following were exact:
model and optimizer state; all three global RNG states; history; all eight
canonical BEST payloads; selected epochs/scores; schedule identity/progress;
sampler order; generated mask rows; and recorded next-epoch draws. This checks
the actual trainer-hook ordering, not only a standalone checkpoint helper.
It does not substitute for full-model CUDA recovery/admission evidence.

Durable test command:

```text
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_block_integration tests.test_meaningful_blocks tests.test_meaningful_resume -v
```

Observed: **8 tests passed in 7.863 seconds**, including all twenty methods'
current-Flat zero-start train/eval comparison. The historical and trainer-hook
probes above were additional read-only inline checks; persisting them as durable
regressions is recommended.

After the grouping dtype correction, additionally ran:

```text
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_grouping tests.test_meaningful_block_integration tests.test_meaningful_blocks tests.test_meaningful_resume -v
```

Observed: **31 tests passed in 8.699 seconds**. An independent full-model
float64 probe for each of `capsule_dynamic_routing`, `slot_attention`, `otke`,
and `capsule_variational_bayes` now passes in both train and eval: exact initial
predictions/global Torch RNG, one causal scan, exact raw shared gradients
(maximum difference zero; tolerance `atol=1e-10, rtol=1e-8`), and strict
canonical-builder checkpoint reload.

## Finding, limitations, and nonblocking follow-up

1. **Resolved medium finding:** grouping's `_GroupingBase.forward`
   originally converted inputs unconditionally with `.float()`. A whole-model
   `.double()` therefore failed at LayerNorm against float64 parameters. This
   prevents the planned float64 integration control for those methods, though
   the fixed cfg84 FP32 probe passes. Its owner added parameter-dtype-aware
   handling: float64 parameters retain float64 computation; ordinary execution
   and disabled-autocast statistical routing retain float32. The full-model
   four-method retest above confirms the original failure is corrected.
2. Runtime OSRAM hook combinations (`read_node`, `write_node`, completion/read
   callbacks) remain an existing low-level surface. The fixed model/config
   cannot enable them with the meaningful flag; adding a direct-backbone guard
   analogous to decision correction would be defensive hardening, not a found
   failure of the authorized screen.
3. Existing miss0-parity failures and the pytest-dependent per-rate-selection
   module are not newly attributed to this integration. Parent independently
   reproduced the two miss0 failures on the frozen baseline; this review did
   not repeat that diagnosis or install pytest.
4. This review does not claim GPU feasibility, full-dataset convergence,
   throughput, memory budget, resume on CUDA, or predictive benefit. Those gates
   remain with the per-candidate preflight and experiment owner.
