# OSRAM additional forty: implementation plan

> Execute under the existing worktree contract. The user authorized autonomous implementation; no formal training is included.

**Goal:** Deliver forty source-grounded, independently selectable readout adaptations, not forty claimed improvements.

**Architecture:** Extend MeaningfulInputAdapter at its existing Flat-input boundary. Preserve original Local skip, OSRAM trajectory, task head, loss, random masks and all old method identifiers. Use existing HeadTokenizer, active_groups and safe_mask; special geometry/dynamics cores may operate on fixed slots instead of tokens.

**Stack:** Existing Python/PyTorch only. No new dependencies.

## Files and ownership

- `docs/osram_new40_*cards.json` and `docs/osram_new40_nine_designs.json`: formulas, sources, adaptation differences and rejected candidates.
- `gcnet_missing_m3/meaningful_new40_registry.py`: immutable identifiers, no model initialization.
- `gcnet_missing_m3/meaningful_input_new40.py`: lazy family dispatch and shared zero-initialized token decoder.
- `gcnet_missing_m3/meaningful_new40_{structure,representation,conditional,geometry,functions,retrieval}.py`: independent mathematical implementations; only their assigned owner edits each file.
- `gcnet_missing_m3/meaningful_input.py`: one optional family registration; existing method code unchanged.
- `tests/test_meaningful_new40.py`: catalog coverage plus reuse of the existing shared contract. No separate general audit suite per method.
- `experiments/osram_new40_20261004/`: source catalog and single-method runner reusing existing audited cfg84 training functions. Not an automatic forty-run queue.

## Sequence

- [ ] Finish and read all source cards; reconcile exactly forty unique core mechanisms against old40 and withdrawn methods. Same hypothesis is allowed; renamed computations are not.
- [ ] Add the shared catalog test before implementation. Assert the new catalog exists, has forty distinct IDs disjoint from old40, and all factory methods construct. Run with `/home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest tests.test_meaningful_new40`; confirm the missing catalog causes the expected failure.
- [ ] Implement family cores with the fixed external contract `forward(local,evidence,active,availability) -> (local,evidence)`. Final correction decoders start at zero. No random forward operations or persistent per-example state. Inner derivatives needed by particular algorithms must work under the existing eval no-grad context.
- [ ] Register the family by extending `INPUT_FAMILIES` only. The existing model's fork_rng initialization remains responsible for unchanged backbone initialization. Old checkpoint loading with the feature disabled stays strict.
- [ ] Run the existing shared input contract: initial identity, dirty inactive inputs, zero padding/first step, finite backward, and parameter updates. Run existing full-model integration for off-path equality, enabled zero-bridge equality and RNG equality. Fix concrete failures only.
- [ ] Implement catalog validation and a single-candidate entry point delegating to the existing audited runner. Preserve required provenance, snapshot, data manifest, healthy GPU UUID and readiness checks. Do not bypass them to call code runnable.
- [ ] Record measured parameter counts and actual verification scope. No W-F1 without a training result; no GPU execution claim from CPU unit tests. Training remains on biggpu excluding physical GPU4, and capped at sixty distinct methods overall.
- [ ] Run `git diff --check`; stage only this task's files; commit with Lore trailers and push the current branch to `github`, never upstream `origin`.

## Acceptance

Exactly forty source-backed registered implementations, no placeholders or aliases. Every ID resolves through the real model entry point. A shared test pass is code correctness evidence, not performance evidence. If a method cannot retain its stated core within the interface, replace it with another checked method rather than quietly simplifying it.
