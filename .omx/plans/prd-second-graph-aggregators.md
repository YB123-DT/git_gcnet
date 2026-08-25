# PRD: GCNet Second-Layer Aggregators

Chinese mirror: [prd-second-graph-aggregators.zh.md](prd-second-graph-aggregators.zh.md). English mirror: [prd-second-graph-aggregators.en.md](prd-second-graph-aggregators.en.md).

## Objective

Determine whether a learnable general set aggregator or a robust medoid estimator improves missing-modality IEMOCAPSix performance when replacing only GCNet's second GraphConv add aggregation.

## User stories

### US-001 — Faithful aggregation modules

As a researcher, I need GenAgg and Soft Medoid implemented from traceable primary sources so the mechanism and attribution are defensible.

Acceptance criteria:

- GenAgg implements augmented f-mean and explicit inverse consistency.
- Soft Medoid implements cardinality-scaled neighborhood medoid weights.
- Torch 1.8/PyG 2.0.1 forward and backward are finite.
- No dependency is added.

### US-002 — Controlled GCNet integration

As an experimentalist, I need the selector to replace only both `conv2` aggregators so all other GCNet mechanisms remain controlled.

Acceptance criteria:

- Default `add` is exactly legacy-equivalent in RNG, state keys, parameters, output, and gradients.
- Both candidate branches change only `conv2`.
- First-layer `--graph-conv-variant` remains `original` in the experiment.

### US-003 — Nonduplicative locked execution

As the project owner, I need the experiment to reuse existing controls and avoid repeated smoke and Original runs.

Acceptance criteria:

- Runner schedules exactly 12 first-wave candidate jobs on four GPUs with three workers each.
- No Original child process is created.
- Existing Original archives are validated by task key and mask SHA.
- Completed first-wave jobs are reused during formal continuation.

### US-004 — Evidence-backed decision

As a paper author, I need a paired report that can accept or reject each module without overstating novelty.

Acceptance criteria:

- Report includes every task, rate mean, seed macro, parameter count, runtime, and provenance.
- Predeclared gate is applied without rounding.
- Failed candidates are recorded and stopped; passed candidates continue to the full locked grid.

## Out of scope

Mask-conditioned aggregation, temperature search, parameter-matched control, edge attention, learned adjacency, new graph layers, new reconstruction objectives, other datasets, and rerunning Original.

