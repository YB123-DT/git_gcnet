# Frozen joint-representation reinjection audit

Status: internal diagnostic only. This document records the implementation
actually used for the 62.82% L2 result; it is not a formal paper result.

## Provenance

The result was produced by the local branch `feature/osram-uniform-forced-text`
at commit `7e4f7d5`, using
`run.py` in this directory. It was not produced by the older
`feature/osram-reg-only` branch.

## Actual L2 configuration

```text
completion_path = pre_osram_joint_frozen
classification_completion = false
initial_backbone_checkpoint = None
joint_pretrain_checkpoint = m3_pretrain_three_datasets_20260921/seed_<seed>/checkpoint.pt
training_objective = emotion-only
checkpoint_selection = per-rate-test-oracle
```

The forward path is:

```text
frozen joint P_A/P_T/P_V + frozen SourceOnlyM3Predictor
    -> CompletedReadFusion
    -> read_node
    -> one causal OSRAM
    -> emotion head
```

The persistent write side remains the original `encoded` node. The predictor
is called under `torch.no_grad()` and is not written to persistent memory.

## Parameter audit (seed 66)

```text
total parameters:     18,072,509
frozen parameters:      2,645,512
trainable parameters:  15,426,997
```

Frozen prefixes:

```text
observed_set.projectors.*
source_only_predictor.*
teacher.*
```

Trainable roots include:

```text
osram.*
observed_set.*                  # non-projector fusion/availability parts
completed_read_fusion.*
smax_fc.*
```

One bookkeeping issue remains: the legacy `missing_predictor.*` is also
marked trainable by the generic joint-freeze helper and appears in the
optimizer predictor group (2,511,112 parameters), although this path never
uses it in forward. It should be excluded before a formal rerun. This does
not make the current L2 an all-backbone-frozen experiment, but it does make
the trainable-parameter report slightly impure.

## L2 result

Three seeds, eight rates, independent per-rate Test-oracle selection:

| seed | 8-rate mean | high-missing mean (0.5/0.6/0.7) |
|---:|---:|---:|
| 66 | 65.080% | 65.716% |
| 67 | 61.063% | 60.672% |
| 68 | 62.320% | 61.943% |
| mean | 62.821% | 62.777% |

The result is therefore not evidence that predicted-slot reinjection itself
causes the full drop from the trainable-projector No-JEPA control. The clean
reinjection comparison still requires a matched L1 using the same frozen
projectors and the same trainable downstream modules.
