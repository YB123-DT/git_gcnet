# GCNet BiLSTM 2×2 Factorial Ablation Design

Date: 2026-08-25

## Objective

Determine whether GCNet's recurrent context encoders help or obscure graph
propagation under missing modalities. The experiment must separately identify
the effects of the pre-graph BiLSTM and the two post-graph branch BiLSTMs,
rather than treating all recurrent layers as one indivisible component.

## Factorial arms

The experiment has two binary factors:

- pre-graph context: `bilstm` or `linear`;
- post-graph branch context: `bilstm` or `linear`.

This produces four arms:

| Arm | Pre-graph | Post-graph temporal/speaker branches |
|---|---|---|
| `original` | BiLSTM | BiLSTM |
| `no_pre_bilstm` | utterance-wise Linear | BiLSTM |
| `no_post_bilstm` | BiLSTM | utterance-wise Linear |
| `no_all_bilstm` | utterance-wise Linear | utterance-wise Linear |

The primary graph operator is official GCNet `RGCNConv`. CP-LECC is not mixed
into this first factorial experiment; its completed eight-rate grid remains a
separate result. If the factorial identifies a useful recurrent setting, the
same setting can subsequently be crossed with CP-LECC in a distinct A/B.

## Exact architecture changes

### Pre-graph replacement

Original:

```text
[T,B,D_in] -> 2-layer bidirectional LSTM(hidden=D_e) -> [T,B,2D_e]
```

Linear ablation:

```text
[T,B,D_in] -> Linear(D_in,2D_e) -> [T,B,2D_e]
```

The Linear is applied independently to every utterance. It preserves the graph
input dimension and a learned feature projection while removing all temporal
recurrence.

### Post-graph replacement

Before the post-graph context module, each graph branch already has
`D_h = 2D_e + graph_hidden_size` features per utterance.

Original:

```text
[T,B,D_h] -> 2-layer bidirectional LSTM(hidden=D_h)
            -> [T,B,2D_h] -> existing Linear(2D_h,D_h) + ReLU
```

Linear ablation:

```text
[T,B,D_h] -> Linear(D_h,2D_h)
            -> [T,B,2D_h] -> existing Linear(2D_h,D_h) + ReLU
```

Both temporal and speaker branches use the same selected mode but retain
independent parameters, exactly as their original BiLSTMs do.

## Shared-module and initialization fairness

Every arm instantiates both the BiLSTM and utterance-wise Linear at each site;
the factor only selects which path participates in `forward`. A bypassed module
must receive no gradient. This keeps the four cells structurally aligned and
prevents different module-construction orders from changing later parameters.

Adapter initialization is performed from a forked CPU RNG state and restores
the state immediately afterwards. Consequently, the on/on cell's official
GCNet parameters and RNG state remain bitwise equal to an unmodified Original
constructed with the same seed. Tests compare every official/common named
parameter across arms after reseeding.

All four factorial cells therefore have the same stored total parameter count,
but different active parameter counts. Reports must include both. For official
GCNet dimensions and Original graph propagation, the active counts are:

| Arm | Active parameters |
|---|---:|
| `original` | 34,140,166 |
| `no_pre_bilstm` | 29,782,166 |
| `no_post_bilstm` | 15,110,166 |
| `no_all_bilstm` | 10,752,166 |

The difference in active capacity is an explicit limitation of a removal
ablation. A later context-free parameter-matched MLP sensitivity control may
test capacity separately, but it is not mixed into the primary 2×2 matrix.

## Frozen components

The ablation must not change:

- input features or modality masking;
- stage-aware train/validation/test mask bundles;
- temporal and speaker graph topology and relations;
- first `RGCNConv` and second `GraphConv`;
- MatchingAttention behavior;
- temporal/speaker branch addition;
- classifier, reconstruction head, losses, optimizer, or best-validation-epoch
  selection;
- hidden sizes, graph windows, batch size, learning rate, or epochs.

## Dataset protocol

Run all four released datasets:

| Dataset | Task/metric | Split |
|---|---|---|
| IEMOCAPFour | 4-class weighted F1 | fold 5 |
| IEMOCAPSix | 6-class weighted F1 | fold 5 |
| CMUMOSI | non-zero binary weighted F1 | official train/val/test split |
| CMUMOSEI | non-zero binary weighted F1 | official train/val/test split |

For every dataset and arm, run missing rates `0.0, 0.1, ..., 0.7` and seeds
`66,67,68,69,70`, using 100 epochs and the same immutable stage-aware bundle for
all four arms of a dataset/rate/seed cell.

The complete screening matrix is
`4 datasets × 4 arms × 8 rates × 5 seeds = 640` jobs. Four V100 GPUs may each
host at most three concurrent jobs. IEMOCAP uses fold 5 to remain comparable to
the current graph experiments; this is explicitly a fold-5 screening result,
not a five-fold LOSO claim. IEMOCAPFour and IEMOCAPSix share the same underlying
corpus and are not counted as independent external replications.

## Operational gate

Before launching the full matrix:

1. Run 16 short smoke jobs: four arms × four datasets × rate 0.7 × seed 66.
2. Run the first reusable 96 formal jobs: four arms × four datasets × rates
   `{0.0,0.7}` × seeds `{66,67,68}` at 100 epochs.
3. Repeat one identical ERC cell and one identical sentiment cell to estimate
   the CUDA run-to-run noise floor.

These gates are operational, not performance-selection gates. All arms continue
to the remaining formal matrix if:

- every job reaches 100 epochs with return code zero;
- every job saves exactly one non-smoke archive;
- output shapes and metric extraction are valid;
- classification arms predict all expected IEMOCAPSix classes;
- paired cells have identical mask bundle hashes;
- no NaN/Inf or active lock remains.
- the on/on cell meets a pre-recorded reproduction tolerance;
- repeated-run W-F1 drift is below the practical-effect threshold 0.005.

## Required tests

1. CLI accepts the four ablation arms and records them in saved provenance.
2. Each arm constructs the expected recurrent/linear modules at all three sites.
3. Every arm returns the same logit, reconstruction, and hidden tensor shapes.
4. The Linear replacements are utterance-wise: changing another timestep cannot
   affect their direct pre-graph/post-graph replacement output.
5. Shared parameters are bitwise identical across arms for the same seed.
6. Stored total and active parameter counts are recorded and mechanically checked.
7. Existing Original, MPFiLM, CP-LECC, mask-bank, and runner tests remain green.
8. The generalized runner builds exactly 640 formal jobs and rejects provenance
   drift or partial artifacts.

## Analysis

For each dataset and rate report mean ± sample standard deviation over five
seeds. Do not pool raw ERC and sentiment metrics. First compute every contrast
within dataset and paired seed. Report the three factorial contrasts against
Original:

- pre-graph BiLSTM effect: `Original - No-Pre`;
- post-graph BiLSTM effect: `Original - No-Post`;
- all-BiLSTM effect: `Original - No-All`.

Also report the interaction contrast
`(Original - No-Pre) - (No-Post - No-All)`, paired-seed win counts, paired
t-tests, Wilcoxon tests, parameter counts, collapse diagnostics, and per-dataset
macro averages across rates. With five pairs, the smallest attainable two-sided
exact Wilcoxon p-value is 0.0625, so inference emphasizes paired effect size,
confidence interval, and sign consistency rather than `p<0.05`.

## Interpretation boundary

This is an ablation of recurrence, not a proposed replacement architecture.
A Linear arm outperforming Original would show that the removed BiLSTM is not
helpful under that protocol; it would not by itself establish the Linear as a
novel method. CP-LECC is crossed with the winning context setting only after
this factorial result is complete.

Removing BiLSTM recurrence does not make the model causal because graph edges
still use `windowf=2`. The pre-graph factor also removes recurrent implicit
completion across neighboring utterances, so its effect combines contextual
encoding and missingness propagation. The post-graph factor jointly controls
both graph branches and is not retrospectively split into separate factors.
