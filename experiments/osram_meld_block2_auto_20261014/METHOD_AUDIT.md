# MELD Block2 finite screen — implementation audit

Audit date: 2026-10-14. Source inspected: `e0bb6b4`, branch
`feature/ecerc-features`. This batch reuses six existing implementations plus
original Flat; it introduces **zero new research mechanisms**. Historical MOSI
experiments, prior source cards and the global method registry already contain
these exact mechanisms. The B1–B6 prefixes identify the finite MELD comparison,
not novelty, additional literature quota or different algorithms.

## Actual routing and write scope

All six names appear in `gcnet_missing_m3.meaningful_input.INPUT_METHODS`.
`OSRAMBackbone` sets `meaningful_input_mode` from that registry and instantiates
the same `MeaningfulInputAdapter` under `torch.random.fork_rng(devices=[])`.
The adapter returns corrected **inputs** to the original Flat MLP. The original
Local Skip still receives `local`, independently of `readout_local`.
Memory scanning, queries, writes, task head and loss are outside these cores.

| Screen ID | Actual core / local implementation | Flat-side Local update | Base/Gap update | Identity initialization |
|---|---|---|---|---|
| Flat | `osram.py`: original Flat1600 MLP and Local Skip | No Block2 correction | No Block2 correction | Original Flat final projection starts zero |
| B1 graph matching | `meaningful_new40_structure.py:GraphMatching` | No; Local is conditioning | Yes; paired Base/active Gap graphs | Shared value decoder weights/bias zero |
| B2 dual attention | `meaningful_input_attention.py:AttentionInputAdapter`, `DualSymbolicAttention` | Yes | Yes | Local and all eight memory output maps zero |
| B3 MBT | `meaningful_input_representation.py:RepresentationInputAdapter`, `_MultimodalBottleneck` | Yes | Yes | Local and all eight memory decoders zero |
| B4 original NPS | `meaningful_new40_conditional.py:TokenReadout`, `NeuralProduction` | **No**; this exact variant has no `local_bridge` | Yes | Eight memory bridges zero |
| B5 TokenLearner/Fuser | `meaningful_input_representation.py:RepresentationInputAdapter`, `_TokenLearnerFuser` | Yes | Yes | Raw decoders zero; token-axis mixer Xavier |
| B6 learned RPCA | `meaningful_input_structured.py:_LearnedRobustPCA` | No | Yes, when at least two evidence roles active | `gamma=0`, so `Y-tanh(gamma)*S=Y` |

B1 runs four graph updates per Base/Gap pair: within-graph edge messages,
bidirectional cross-graph mismatch, then simultaneous GRU node updates.
Base corrections are averaged over participating pairs. B2 runs separate
sensory attention and relation-valued attention with content-derived symbols,
then concatenates both paths. B3 retains role streams and performs two
synchronous consensus exchanges through four bottleneck tokens, excluding
absent roles from proposal averaging. B4 executes three deterministic hard
rule/primary/context selections with straight-through gradients and dispatches
only the selected rule. B5 learns distinct analysis and synthesis maps around
four compressed tokens, a latent Transformer and token-axis mixing. B6 uses
detached deterministic rank-one SVD initialization and fourteen learned
simultaneous scaled factor updates, followed by signed sparse reweighting.

The source TokenFuser token mixer normally starts zero. This existing adapter
uses Xavier because a zero mixer followed by a zero raw decoder would make a
dead serial branch. B6's detached SVD, bounded step sizes and Gram floors are
existing declared adaptations. None is changed or counted as a new method here.

## Shared mask, history and gradient boundary

Wrapper inputs have Local `[T,B,256]`, Base `[T,B,1024]`, Gap `[T,B,3,1024]`
and availability `[T,B,3]`. The configured producer is causal 8x64, so cores
consume only the forward512 half. Backward512 halves are carried through.
The wrapper sanitizes padding with `torch.where`; a Gap is active exactly when
the utterance is valid and its modality is missing. Active-head compaction
excludes observed Gap slots from attention, graph/stream denominators and
decomposition. The seven nonempty binary availability patterns are valid;
all-missing valid utterances are rejected.

Only rows after the first valid utterance with nonzero already-read evidence
run the core. The first valid utterance, all-history-zero ablation and padding
bypass correction. No state persists between utterances and no time ordering
is invented between head tokens. Tests poison inactive slots with NaN/Inf,
verify unchanged outputs, strict zero inactive Gaps/padding, zero backward
halves, initial identity and no-history bypass after learning.

Zero decoders/gamma initially block upstream core gradients. The test takes
three task-shaped random linear-loss SGD steps and checks finite gradients,
nonzero upstream gradients after opening the bridge, nonzero memory changes
and the exact Local-update metadata. It does not require every unselected NPS
rule or absent MBT role to receive a gradient. CPU contract checks are evidence
of interface and gradient connectivity, not accuracy or GPU capacity.

## Existing source and dedup evidence

| ID | Prior local audit / registry entry | Primary method and prior code pin |
|---|---|---|
| B1 | `docs/osram_new40_structure_cards.json`; global registry exact `graph_matching_base_gap_pairs` | [Graph Matching Networks](https://proceedings.mlr.press/v97/li19d.html); DeepMind `f5de0ede8430809180254ee957abf36ed62579ef` |
| B2 | `osram_meaningful20_round2_20261004/attention_candidates.json`; existing attention family | [Dual Attention](https://proceedings.mlr.press/v267/altabaa25a.html); Awni00/dual-attention `dce218cbf5ec9aa7f90687c1323050a1fba17966` |
| B3 | `osram_meaningful20_round2_20261004/representation_candidates.json`; existing representation family | [MBT](https://arxiv.org/abs/2107.00135); Scenic `8c113c501c9f700b69899c55a69e65bb46727da6` |
| B4 | `docs/osram_new40_conditional_cards.json`; global registry exact `conditional_new_07_neural_production` | [Neural Production Systems](https://arxiv.org/abs/2103.01937); prior author source `anirudh9119/neural_production_systems/synthetic/RuleNetwork.py` |
| B5 | `osram_meaningful20_round2_20261004/representation_candidates.json`; existing representation family | [TokenLearner V1.1/Fuser](https://arxiv.org/abs/2106.11297); same Scenic pin as B3 |
| B6 | `osram_meaningful20_round2_20261004/structured_candidates.json`; global registry exact `learned_robust_pca_evidence_decomposition` | [Learned Robust PCA](https://arxiv.org/abs/2110.05649); caesarcai/LRPCA `d241beef37f855a5bab298f58861611f8d945f7b` |

Source attribution above is inherited from existing inspected source cards;
this task audited actual local implementations and routing, without claiming
a fresh upstream reproduction. NPS's old card uses a mutable author link;
the **local implementation snapshot** is the executable authority for this
reuse screen. B4 must not silently switch to `neural_production_local` or its
width variant, which would change the recorded Local write scope.

## Configuration contract

`METHODS` has exactly `Flat`, `B1`, `B2`, `B3`, `B4`, `B5`, `B6`; every record exposes
`name`, `block`, `sources`, `updates_local`, `initnotes`.
`build_config(base_dict, method, output, overrides=None, epochs=50)` returns
a deep-copied plain TrainConfig dictionary. The caller owns `output` and
feature paths; no non-TrainConfig provenance/output fields are added.

The historical `BASE_CONFIG.json` retains dropout0.5, batch32, Adam1e-3,
50epochs, seed66, cyclic0.0–0.7, GCNet random-mask-v1, causal8x64,
write-step0.6, Flat1600 and the original Local Skip. `training_objective` is
`emotion-only`; historical inert `jepa_weight=0.1` remains copied as recorded,
not interpreted as an active JEPA loss. The builder sets per-rate Test-oracle
ACC selection explicitly and enables `preserve_selection_artifacts=True` so
selected checkpoints can retain their winning-mask predictions and UA/W-F1.
All scores from this selection remain **INTERNAL DIAGNOSTIC ONLY**.

First-round structure is fixed. Overrides permit only `learning_rate`,
`weight_decay`, `dropout`, with finite valid values. Every training run,
including each Optuna trial, retains 50epochs; 20 is the trial count cap.
Unknown method IDs,
protocol mismatches, extra interventions and weight-initialized starts are
rejected. Candidate eligibility and the at-most-two/20-trial queue cap belong
to the runner/queue, not this registry.

## Verification

TDD red: `python -m unittest tests.test_meld_block2_methods.RegistryTests.test_registry_module_exists -v`
failed with `Missing finite MELD Block2 registry` before `methods.py` existed.

CPU verification is run in an isolated biggpu directory with
`CUDA_VISIBLE_DEVICES=""`, current `gcnet_missing_m3` source, and
`/data2/yb/conda_envs/cmerc-repro/bin/python`:
`python -m unittest tests.test_meld_block2_methods -v`.
Final isolated directory:
`/data2/yb/remote_experiments/meld_block2_methods_cpu_k5p5W2`.
The complete six-test suite passed with exit0, no skips:

```text
test_real_wrapper_masks_identity_and_upstream_learning ... ok
test_config_preserves_actual_baseline_and_copies_nested_values ... ok
test_exact_seven_existing_mechanisms_and_local_update_metadata ... ok
test_hpo_changes_only_three_training_knobs_and_budget ... ok
test_protocol_mismatch_or_nonregistry_method_is_rejected ... ok
test_registry_module_exists ... ok
Ran 6 tests in 4.683s
OK
```

The mechanism test iterates all six real adapters and all seven legal masks.
Local suite: five configuration/registry tests pass; the PyTorch mechanism
test is skipped because local Python has no Torch. Python compilation and
`git diff --check` pass. An additional direct build using the actual historical
`BASE_CONFIG.json` verified all seven configurations preserve dropout0.5,
inert jepa_weight0.1 and 50epochs, adding only the two new trainer fields
`checkpoint_selection_metric` and `preserve_selection_artifacts`.

No GPU smoke or formal training is launched by this slice. Baseline retraining need not be bitwise
identical to the archived seed66 score; frozen evaluation replay and fresh
training reproducibility have different evidence boundaries.
