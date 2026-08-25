# PRD: GCNet Second-Layer Aggregators

Chinese mirror: [prd-second-graph-aggregators.zh.md](prd-second-graph-aggregators.zh.md). English mirror: [prd-second-graph-aggregators.en.md](prd-second-graph-aggregators.en.md).

## Objective

Determine whether GenAgg, Soft Medoid, SSMA cross-neighbor mixing, or custom RTDR diagonal relation-transition routing improves missing-modality IEMOCAPSix performance while preserving controlled GCNet protocols and inherited Original evidence.

## User stories

### US-001 — Faithful aggregation modules

As a researcher, I need GenAgg, Soft Medoid, and SSMA implemented from traceable primary sources, while RTDR is labeled as a custom hypothesis, so every mechanism and attribution is defensible.

Acceptance criteria:

- GenAgg implements augmented f-mean and explicit inverse consistency.
- Soft Medoid implements cardinality-scaled neighborhood medoid weights.
- SSMA implements the core no-attention fixed-signal FFT mixing and full compressor.
- RTDR is not attributed to MrMP and explicitly changes only the two-hop relation-transition mask.
- Torch 1.8/PyG 2.0.1 forward and backward are finite.
- No dependency is added.

### US-002 — Controlled GCNet integration

As an experimentalist, I need each selector to expose one explicit controlled change.

Acceptance criteria:

- Default `add` is exactly legacy-equivalent in RNG, state keys, parameters, output, and gradients.
- GenAgg, Soft Medoid, and SSMA change only both `conv2` aggregation primitives.
- RTDR decomposes the existing first-layer relation messages without changing weights, normalization, edges, relation IDs, or parameters, and changes only the second-hop transition mask.
- First-layer `--graph-conv-variant` remains `original` in every experiment.

### US-003 — Nonduplicative locked execution

As the project owner, I need the experiment to reuse existing controls and avoid repeated smoke and Original runs.

Acceptance criteria:

- Runner schedules Phase A (GenAgg + Soft Medoid) as 12 jobs and Phase B (SSMA + RTDR) as 12 jobs, each on four GPUs with three workers per GPU.
- No Original child process is created.
- Existing Original archives are validated by task key and mask SHA.
- Completed first-wave jobs are reused during formal continuation.

### US-004 — Evidence-backed decision

As a paper author, I need a paired report that can accept or reject each module without overstating novelty.

Acceptance criteria:

- Report includes every task, rate mean, seed macro, parameter count, runtime, and provenance.
- Predeclared gate is applied without rounding.
- Failed candidates are recorded and stopped; passed candidates continue to the full locked grid.
- After all verified jobs finish, completed code/results are synchronized locally, committed with provenance, and pushed to `YB123-DT/git_gcnet`; incomplete jobs are excluded from completed-version folders.

## Out of scope

Mask-conditioned aggregation, temperature search, edge attention, learned adjacency, new reconstruction objectives, other datasets, rerunning Original, naive ego-neighbor reparameterization, and stateless clipping mislabeled as source-faithful Centered Clipping. A parameter-matched SSMA control is allowed only after SSMA passes its first gate.
