# GCNet Completed Versions Repository Layout

## Goal

Present the completed GCNet experiments as one readable repository with one
shared runtime and one small directory per method. The public-facing layout
must not expose unfinished candidates as completed methods.

## Included scope

Only these completed paths are included:

1. Original GCNet;
2. Missing-Pattern FiLM RGCN, including the faithful edge-wise diagnostic;
3. CP-LECC RGCN;
4. mask-conditioned Sequence AFF.

The repository records only the IEMOCAPSix fixed-fold-5 evidence currently
available for these paths. It must describe this as a fixed fold, never as a
five-fold cross-validation result.

G1U/G1S, D0/D1, BiLSTM factorial ablation, ReMasker, GenAgg, Soft Medoid,
PConv and all other unfinished candidates are excluded. They remain in local
research branches until their locked experiments finish.

## Target tree

```text
README.md
environment/
  OFFICIAL_ENVIRONMENT.md
  requirements.txt
common/
  gcnet/
versions/
  original/
    variant.py
    config.json
    README.md
  mpfilm/
    variant.py
    config.json
    README.md
  cp_lecc/
    variant.py
    config.json
    README.md
  sequence_aff/
    variant.py
    config.json
    README.md
results/
  iemocap6/
    fold5/
      original/
      mpfilm/
      cp_lecc/
      sequence_aff/
tests/
run.py
```

## Code boundary

`common/gcnet/` owns code that is identical across all four arms: data loading,
mask-bank handling, graph construction, losses, metrics, optimizer setup,
checkpoint/archive handling and the unchanged GCNet trunk.

Each `versions/<name>/variant.py` owns only its replacement mechanism and a
small registration function. `config.json` owns the locked switches for that
arm. The root `run.py` resolves `--version`, loads the configuration, registers
the selected variant and delegates to the shared trainer.

The Original variant is a no-op registration. It must not instantiate method
parameters, consume extra RNG or change Original state-dict keys.

MPFiLM keeps its linearized and faithful-edge-wise diagnostic modes under one
top-level version. CP-LECC keeps its completed full arm and documented
controls. Sequence AFF changes only the speaker/temporal branch fusion.

No new package dependency is allowed. The shared environment remains the
official Python 3.8, Torch 1.8 and PyG 2.0.1-compatible environment used by the
formal runs.

## Results boundary

The Git repository stores compact evidence only:

- Markdown conclusions;
- JSON/CSV task summaries when they exist;
- source commit, dataset, fold, seeds, requested missing rates, mask/provenance
  hashes and the external artifact root;
- explicit missing cells where an arm was not run.

It does not store feature archives, full datasets, checkpoints, raw per-epoch
NPZ files, mask banks or full training logs. CP-LECC must report only the rates
actually completed; absent rates must not be synthesized.

Every result directory contains a README that states whether the evidence is a
full 8-rate by 5-seed table, a narrower diagnostic, or an inherited Original
control.

## Migration and provenance

The organized branch is built in an isolated worktree from the committed
research source. A machine-readable source map binds every moved module and
result to its original branch, commit and path. Moving code must not erase the
historical origin of a result.

The existing GitHub branches remain untouched until the organized tree passes
verification. Remote branch cleanup is a separate final operation performed
only after the new main branch is verified. Local branches remain as the full
research archive.

## Verification

The layout is accepted only when:

1. all four versions import under the official Python environment;
2. Original forward output, parameters, state-dict keys and RNG behavior match
   the locked Original implementation;
3. MPFiLM, CP-LECC and Sequence AFF focused unit/integration tests pass;
4. `run.py --version <name> --help` resolves all four arms without source edits;
5. every compact result agrees with its committed source summary;
6. no unfinished method name or implementation occurs under `versions/` or
   `results/`;
7. no tracked blob exceeds the repository artifact policy;
8. the GitHub branch SHA matches the locally verified commit.

## Failure handling

If extracting a shared component changes numerical behavior, it stays as a
version-owned override until an equivalence test proves it can be shared. The
layout must prefer small duplication over falsely claiming two non-equivalent
paths are common.

Remote historical branches are not deleted when any verification item fails.
