# Engineering interfaces after temporal encoding: residuals, dimensions, initialization and normalization

2026-10-22. Read-only code and paper review. OSRAM source: feature/ecerc-features at a88bffbfc01a5de209d95869abed61eaba22456d. No model edits, training, GPU inference or SSH jobs were performed. The user's refinement prioritizes residual and dimensional details.

## Finding

Several reference implementations retain separate feature blocks by concatenation at module boundaries, while using residual additions inside modules. Original Nested and V3 residual often decode a new representation into the old evidence coordinates, add it to those coordinates, and pass the mixture through the existing Flat. This changes the representation constraints and gradient path.

That is a verifiable engineering distinction, not an identified sole cause of persistent non-improvement. Direct replacement, concatenation, wider modules and frozen-backbone experiments already exist in this repository.

## Concrete dimensions and connections

| Implementation and entry | Dimensions | Fusion and initialization |
|---|---|---|
| DialogueGCN, graph-enabled IEMOCAP script | Text100 → BiLSTM200 → graph100; concat300 → hidden100 → 6 classes | Original contextual state and graph state are concatenated; no explicit zeroed graph exit; classifier Linear/ReLU/Dropout/Linear; no feature LN/BN here |
| MMGCN, official run.sh | Each modality200 → graph100; concat300 each, then 900 jointly → 6/7 classes | Initial-state mixing inside GCNII; final original/graph concatenation; nonzero graph weights; dropout and adjacency normalization, not feature LN |
| M3NET, README hyper/concat_DHT/avl | IEMOCAP nodes512 or MELD nodes1024 → common512; parallel graph512+hypergraph512 → concat1024; three modalities → 3072 | Add inside pairwise graph layers; concatenate branches at exit; optional raw-node bypass is OFF in the training CLI; no LN after internal addition |
| MulT, all three target streams | Each modality30; two directional outputs concatenate to60 per target; three final target states concatenate to180 | Pre-LN attention/FFN residuals, d→4d→d FFN, final residual MLP180; nonzero Xavier weights with zero biases |
| MFN | Sum of modal hidden widths H; two cell-state times concatenate to2H; candidate memory d_mem; final H+d_mem | Final hidden states and recurrent cross-modal memory concatenate; ordinary parameter initialization; zero initial states are not zero output weights |

Pinned code: [DialogueGCN](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/DialogueGCN/model.py#L789), [MMGCN](https://github.com/hujingwen6666/MMGCN/blob/85732d984f70c1c84dd47c81aa97f1271397b899/model_GCN.py#L435), [M3NET](https://github.com/feiyuchen7/M3NET/blob/4d0fa88cd4f92c72b053527e877763f54e29ccb2/model_hyper.py#L153), [MulT](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/models.py#L108), [MFN](https://github.com/pliang279/MFN/blob/b0453fb6e21b581c796246a0c32a6551f2028c1b/test_mosi.py#L183).

These are entry-specific dimensions, not universal paper hyperparameters. DialogueGCN requires enabling graph-model. MMGCN defines an unused feature_fc in the return_feature path; furthermore its MMGCN constructor hardcodes 64 layers although run.sh requests four. We do not certify that checkout as an exact reproduction of the paper's depth table. M3NET's CLI overrides the constructor's use_residue default; its internal graph additions remain active. MFN widths are configuration-dependent, not inferred from the random search loop.

## OSRAM's actual interface

For standard cfg84:

```text
[Local256; Base1024; Gap-A1024; Gap-T1024; Gap-V1024] = 4352
  → LayerNorm4352 → Linear4352,1600 → GELU → Dropout
  → zero-initialized Linear1600,1600
  + LocalSkip256,1600
  → LayerNorm1600 → original task head
```

Each context has 512 forward coordinates and 512 zero backward coordinates. The adapter has 9,535,104 parameters; default V3 has 335,006. This capacity difference does not establish superiority. Replacing LN4352 with LN2304 is not an equivalent removal of unused coordinates: the zeros participate in normalization statistics and need not remain zero after affine normalization. See [OSRAM](../../gcnet_missing_m3/osram.py#L1165) and [recorded counts](../osram_module_connection_audit_20261022/INITIAL_GRADIENTS.json).

Original Nested splits each role's512 into eight genuine64-dimensional heads and retains eight tokens. Its per-head decoder64→64 reconstructs the512-wide slot. Local has its own256→64 token. In V3, each role/source read512 is projected to64, and the final three-source attention yields ONE64-dimensional context per role, decoded64→512. Both use the number64, but they impose different bottlenecks. [Tokenizer](../../gcnet_missing_m3/meaningful_blocks_common.py#L12), [Nested adapter](../../gcnet_missing_m3/meaningful_input_new40.py#L18), [V3](../../gcnet_missing_m3/source_role_v3.py#L28).

For a fixed role and checkpoint, active V3 corrections obey delta_i=W z_i+b, W of shape512×64. Differences between active outputs lie in a subspace of dimension at most64. Uncentered outputs, or mixing active affine outputs with masked zero outputs, have a safe rank bound65. The64 bound does not apply to the original evidence plus correction, nonlinear Flat outputs, or multiple checkpoints. Width expansion at the decoder cannot restore independent degrees of freedom discarded at its input; this does not prove that64 is insufficient.

## Why add and concat are not interchangeable

With a linear consumer, concatenation permits W_x x+W_z z; addition yields W(x+z)=Wx+Wz. Once summed, the next layer generally cannot distinguish the two operands. OSRAM is nonlinear, but its evidence addition still happens before the classifier can separately inspect original and new coordinates.

Thus V3 must learn a correction expressed in the original read coordinate system. A learned decoder can do this, so it is not inherently wrong. Concatenation permits separate coordinates but increases capacity and optimization/overfitting exposure; it is not guaranteed to help.

There is already a [Nested concat experiment record](../osram_nested_concat_20261011/STATUS.md) with a capacity control. Its [direction review](../osram_nested_concat_20261011/DIRECTION_REVIEW.md) contains an interim snapshot, not a verified final result. Do not present concatenation as an untried cure.

## Stacked zero initialization

The original Flat's final weight W2 is zero, and V3's exit D is also zero. In a from-scratch input-residual route A(x+D z), the first task backward can update W2 but sends zero task gradient to D. After W2 changes, D can learn; after D changes, V3's internal network can learn.

Existing synthetic CPU evidence, with dropout=0 and weight_decay=0:

| Backward | Flat final grad norm | V3 Base decoder | V3 PairMLP |
|---:|---:|---:|---:|
| 1 | 7.639775 | 0 | 0 |
| 2 | 7.064045 | 0.284228 | 0 |
| 3 | 5.017679 | 0.313444 | 0.101972 |
| 4 | 8.913265 | 0.634138 | 0.220265 |

[Existing raw record](../osram_module_connection_audit_20261022/INITIAL_GRADIENTS.json); not rerun this turn. This demonstrates a startup delay rather than permanent disconnection; it does not establish whether the delay contributes to long-run performance. Task-gradient statements do not exclude weight-decay updates. Pretrained Flat, hidden-output residuals and randomly initialized decoders have different startup paths. Zero bias with a nonzero weight, as in MulT, does not disable a branch.

## Normalization and magnitude

MulT's local sublayer has x+F(LN(x)); V3's multi-source self-attention uses LN(x+Attention(x)). A single-source V3 row bypasses that normalization. V3's decoded evidence then passes through Flat input LN and the final LN after Local Skip addition. MulT also has encoder-level normalization, so its complete path is not norm-free.

LayerNorm removes common-coordinate shifts and, ignoring epsilon, positive global rescaling. It does not erase arbitrary directional corrections. Scaling only Memory while preserving Local changes relative coordinates and is NOT uniform rescaling of the complete Flat input. Graph adjacency normalization is not the same operation as feature LN.

For small fixed-weight perturbations, delta_y is approximately J_head J_norm J_adapter delta_C. A residual/input norm ratio is therefore not a decision-contribution percentage. Small changes can flip near-boundary predictions; large changes can fall in locally insensitive directions. There is no evidence here that all additions are erased by LN or that a larger residual is intrinsically better.

## Published gains are conditional

These are within-paper results, not comparisons against OSRAM, and they do not isolate add/concat/LN:

| Published comparison | F1 |
|---|---|
| DialogueGCN Table5, IEMOCAP sequential only → sequential+speaker graph | 55.30 → 64.18 |
| COGMEN Table6, IEMOCAP4 text-only without GNN → full | 81.18 → 81.55 |
| COGMEN same table, ATV without GNN → full | 80.28 → 84.50 |
| MMGCN Table4, IEMOCAP late / gated / MFN / MulT fusion | 64.62 / 64.45 / 62.77 / 62.37 |
| MMGCN same table, MELD | 58.26 / 58.18 / 58.21 / 57.93 |

Primary sources: [DialogueGCN](https://aclanthology.org/D19-1015.pdf), [COGMEN](https://aclanthology.org/2022.naacl-main.306.pdf), [MMGCN](https://aclanthology.org/2021.acl-long.440.pdf). MMGCN comparisons are not certified parameter/compute-matched operator ablations. COGMEN was inspected as a paper only.

Temporal access also differs. DialogueGCN/M3NET retain utterance nodes; V3 reshapes to T*B*4 independent rows and attends to three source tokens. MMGCN's default A/V streams are merely linear before the graph. MFN maintains cross-modal state at word-level within each utterance. Many conversational graph references use future context. Their gains cannot be transferred to causal CHM by matching module names.

## Implications and limits

Multiple OSRAM variants share a constrained exit, evidence-coordinate addition and the existing large Flat consumer. Changing their internal operator may leave that interface largely unchanged. This is a plausible common limitation, not proof that Memory is useless or writing must be faulty.

Counterevidence matters: [MOSEI Source-Role switching](../osram_mosei_source_role_switch_20261021/completed/INTERPRETATION.md) found a positive within-checkpoint calibration effect without beating independently trained Flat. Direct-input, frozen and raw-history trials have also failed in limited settings. [Prior audit](../osram_module_connection_audit_20261022/RESULT.md).

If future comparisons are authorized, isolate one interface constraint with the same core and input rather than switching several mechanisms simultaneously. No such experiment was added here.

Source versions/hashes: [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Metadata: [paper_bank.json](paper_bank.json). Extracted within-paper numbers: [paper_comparisons.csv](paper_comparisons.csv). Model checkpoints, masks, datasets, ongoing jobs and prior results were untouched. Reference performance was not reproduced.
