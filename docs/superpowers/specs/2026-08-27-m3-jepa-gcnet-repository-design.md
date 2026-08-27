# M3-JEPA to GCNet Repository Design

## 1. Decision

The official M3-JEPA checkout remains a read-only upstream reference at
`/data2/yb/paper/M3-JEPA`. The GCNet adaptation is developed in the
`git_gcnet` repository on branch `feature/m3-jepa-gcnet`, with one method per
directory and shared datasets, protocols, and environments.

The upstream reference is pinned to:

- repository: `https://github.com/HongyangLL/M3-JEPA.git`
- branch: `main`
- commit: `6f1e4d4`

The adaptation must record which mechanisms are inherited from M3-JEPA and
which mechanisms are newly designed for complete-modality tri-modal
conversational emotion recognition.

## 2. Goals

1. Keep Original GCNet, PLCI-JEPA, and the M3-JEPA adaptation independently
   readable and runnable.
2. Reuse dataset splits, metrics, result provenance, and the existing GCNet
   environment where compatible.
3. Prevent third-party source code, model weights, raw datasets, and incomplete
   experimental artifacts from being mixed into the method implementation.
4. Make source attribution and comparison controls explicit enough for a paper
   audit.

## 3. Non-goals

This repository-design change does not yet:

- implement BERT/DeBERTa, HuBERT, ViT, LoRA, MMoE, or JEPA training;
- choose between two-stage pretraining and end-to-end joint training;
- download pretrained model weights or raw media;
- start smoke tests or formal experiments;
- modify Original GCNet or PLCI-JEPA behavior.
- introduce artificial missing-modality patterns, dual-source-to-target
  prediction, subset lattices, or missing-modality reconstruction.

Those decisions require a separate method specification because they alter the
scientific hypothesis, compute budget, and comparison protocol.

## 4. Directory Layout

```text
git_gcnet/
├── gcnet/                         # Original GCNet; unchanged
├── gcnet_plci_jepa/               # Existing PLCI prototype; unchanged
├── gcnet_m3_jepa/                 # New adaptation, created during implementation
│   ├── __init__.py
│   ├── encoders.py                # Tri-modal encoder and LoRA interfaces
│   ├── mmoe.py                    # Adapted direction-conditioned predictor
│   ├── jepa.py                    # Complete-modal pairwise JEPA tasks and losses
│   ├── model.py                   # Integration with the GCNet backbone
│   ├── train.py                   # Method-specific training entry point
│   └── README.md                  # Mechanism and execution documentation
├── configs/
│   └── m3_jepa/                   # Versioned experiment configurations
├── tests/
│   └── test_m3_jepa/              # Method-specific regression tests
├── docs/
│   ├── designs/                   # Scientific method specifications
│   └── upstream/
│       └── M3_JEPA_UPSTREAM.md    # Upstream commit and code mapping
└── experiments/
    ├── completed/                 # Reportable completed runs only
    └── incomplete/                # Interrupted or diagnostic artifacts
```

Existing shared folders remain the source of truth for datasets and environment
records. The new method directory must not duplicate them. Fixed missing-mask
banks are outside the scientific protocol of this complete-modality method.

## 5. Ownership and Dependency Boundaries

### 5.1 Read-only upstream

`/data2/yb/paper/M3-JEPA` is used to inspect the original mechanism and source
implementation. No project changes are made there, and it is not copied in full
into `git_gcnet`.

### 5.2 Original GCNet

`gcnet/` remains the comparison baseline. Shared behavior is reused through
stable utilities or imports. Original files are not silently edited to support
the new method.

### 5.3 Adaptation directory

`gcnet_m3_jepa/` owns only the changes required for the new method. Its public
entry point must make the following boundaries visible:

- encoder adaptation;
- cross-modal predictor;
- GCNet integration;
- training-only objectives;
- inference path.

The method must be understandable without reading the upstream M3-JEPA
repository.

### 5.4 Shared protocol

Dataset splits, metrics, seeds, and result serialization remain shared.
Method-specific code may consume these interfaces but may not redefine them
under the same names. Every valid utterance supplies Audio, Text, and Visual
inputs; an input mask is used only for sequence padding, never to manufacture a
missing-modality benchmark.

## 6. Scientific Scope Boundary

The public M3-JEPA implementation exposes single-source, single-target
directional prediction, exemplified by `text2image` and `image2text`. The
complete-modality GCNet adaptation therefore generalizes this mechanism to six
pairwise directions:

```text
A -> T    T -> A
A -> V    V -> A
T -> V    V -> T
```

All three modalities are present in the training example. A directional task
chooses one modality as source and another as prediction target; the third
modality is not treated as missing. The design does not invent `AT -> V`,
`AV -> T`, or `TV -> A` tasks, because those would require a new subset-fusion
predictor rather than a faithful, simple M3-JEPA transfer.

The scientific question is whether M3-style cross-modal latent alignment,
combined with parameter-efficient adaptation of Text, Audio, and Visual
encoders, improves complete-modality emotion representations before GCNet's
temporal and speaker reasoning. It is not a claim about robustness to missing
modalities.

## 7. Provenance Rules

`docs/upstream/M3_JEPA_UPSTREAM.md` must contain:

1. upstream repository and pinned commit;
2. upstream license/header status;
3. file-to-mechanism mapping for `m3-jepa.py` and `MMoE.py`;
4. a table separating copied, adapted, and newly authored mechanisms;
5. any deviations from the paper or incomplete portions of the public code.

Copied code must retain its original copyright and license header. Prefer a
clean reimplementation from the published mechanism when only a small component
is needed. The paper must not claim the inherited M3-JEPA predictor or LoRA
usage as a new core contribution.

## 8. Artifact Policy

The Git repository includes:

- source code and tests;
- lightweight configurations;
- dependency and model-name manifests;
- hashes and provenance metadata;
- compact summaries of completed experiments.

The Git repository excludes:

- pretrained BERT/DeBERTa, HuBERT, and ViT weights;
- raw text, audio, frames, and videos;
- generated feature caches;
- checkpoints and optimizer states;
- transient logs, process manifests, and partial result arrays.

Interrupted runs are retained locally under `experiments/incomplete/` and are
never included in aggregate result tables.

## 9. Branch and Commit Policy

Development occurs on `feature/m3-jepa-gcnet`. The branch is based on the
current organized GCNet research state so that datasets and experiment tooling
remain reusable.

Commits follow the repository Lore protocol. Each scientific change records:

- the hypothesis it tests;
- the preserved protocol constraints;
- rejected alternatives;
- exact verification performed;
- known untested conditions.

Method implementation, performance optimization, and formal results are kept
in separate commits.

## 10. Verification Requirements

Before method implementation begins, repository verification must show:

1. the new branch and worktree exist without modifying the dirty main worktree;
2. Original GCNet and PLCI-JEPA tracked files are unchanged;
3. the upstream commit is recorded and reproducible;
4. no model weights, datasets, feature caches, or large generated artifacts are
   staged;
5. the proposed module can import shared split and metric utilities without
   copying them;
6. repository documentation distinguishes inherited M3-JEPA mechanisms from
   the new GCNet adaptation.
7. complete-modality experiments do not load or generate a natural missing-mask
   bank.

Method-level unit tests and formal experimental tests belong to the subsequent
architecture specification.

## 11. Alternatives Considered

### Git submodule

Rejected because it adds clone, authentication, synchronization, and remote
execution friction while the public M3-JEPA repository currently lacks training
scripts and checkpoints.

### Modify the official M3-JEPA checkout directly

Rejected because GCNet datasets, masks, metrics, and result provenance would be
split across two repositories, making comparisons and paper auditing harder.

### Copy the complete M3-JEPA repository into `git_gcnet`

Rejected because it imports unrelated code, obscures authorship boundaries, and
creates two sources of truth for upstream behavior.

### Keep every method in one shared Python package

Rejected because prior experiments showed that method switches and shared-file
edits make version provenance harder to inspect. One method per directory is
the preferred human-readable boundary.

### Extend M3-JEPA directly to dual-source-to-target prediction

Rejected for this version because the public M3-JEPA implementation is
single-source and single-target. A dual-source predictor would introduce a new
fusion problem and collapse this deliberately simple complete-modality transfer
back into the PLCI/subset-lattice research direction.

## 12. Primary Risk

Replacing frozen pre-extracted features with raw-input BERT/DeBERTa, HuBERT,
and ViT encoders is not a small GCNet module replacement. It requires raw-media
alignment, preprocessing, substantially more GPU compute, and a same-encoder
non-JEPA control. Without that control, any gain could be caused by stronger
encoders or LoRA rather than M3-style alignment. The repository boundary is
therefore established first; the exact joint-training schedule remains a
separate reviewed decision.
