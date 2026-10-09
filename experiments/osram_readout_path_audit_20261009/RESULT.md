# Local/Base/Gap candidate placement audit

INTERNAL DIAGNOSTIC ONLY

Read-only code/result audit at commit 0dea9c7. No training, new inference,
checkpoint intervention, weight change or model-code edit. Scope is the
existing high-score shortlist plus documented replacement controls, not an
exhaustive count of every historical candidate. Unrelated gcnet/model.py
worktree changes are excluded; the audited model is gcnet_missing_m3/model.py.
Scores below use the already reported per-rate Test-oracle W-F1 protocol.

## What actually executed

Let X=[L,B,masked G_A,masked G_T,masked G_V], S=local_skip,
A=emotion_adapter, N=emotion_norm, H=original task head.
Original prediction is H(N(S(L)+A(X))). All formulas suppress padding masks.

| Candidate | Actual readout computation | Original Flat retained? | Memory read/write changed? |
|---|---|---|---|
| Nested / CWN | H(N(S(L)+A(X+decoded_graph(X)))) | Yes; original Local Skip uses unmodified L | No |
| Original NPS | H(N(S(L)+A([L,C+decoded_rules(L,C)]))) | Yes; only Base/Gap adapter inputs changed | No |
| NPS + Local, W96/W256 | H(N(S(L)+A([L+delta_L,C+delta_C]))) | Yes; still no Local Skip correction | No |
| Nested direct zero/random | H(N(S(L)+A(decoded_graph(X)))) | Yes; removes input residual additions only | No |
| XCA / Tucker / Perceiver / Sheaf | H(N(S(L)+A(X)+R(L,C,a))) | Yes; parallel pre-norm residual | No |
| D3-W256 MemoryShiftFilter | H(N(S(L)+A(X)+P(filtered_shift(L,C,a)))) | Yes; unfiltered context still enters A | No |
| Evidence-centered decision correction | sL+dB+dG, from a new decision head | No execution of old A/S/N/H | No |

The formulas describe valid history rows. MeaningfulInputAdapter explicitly
leaves first/empty-history rows unchanged. Gap slots remain availability-masked.
The direct input variant is therefore not even an everywhere replacement of X.

## Two corrections to previous explanations

1. Nested direct did NOT delete Flat. TokenAdapter(residual=False) removes
   additions to the graph-decoded Local/evidence outputs. The receiving
   emotion_adapter, original local_skip and emotion_norm still execute.
2. Perceiver/XCA/Tucker did NOT consume the fused 1600-d Flat representation.
   Their core input is L and active forward Base/Gap evidence. flat_anchor
   supplies output shape and norm diagnostics in MeaningfulReadoutResidual,
   not the core's feature input. Addition at the Flat output is different from
   putting the whole module downstream of Flat.

Code anchors (at audited commit):
- gcnet_missing_m3/meaningful_input_new40.py:20-48: TokenAdapter output modes.
- gcnet_missing_m3/meaningful_new40_structure.py:477-488: Nested/CWN factories.
- gcnet_missing_m3/meaningful_new40_conditional.py:54-77: NPS Local/evidence bridges.
- gcnet_missing_m3/meaningful_input.py:57-93: history guard, fixed slots, masks.
- gcnet_missing_m3/meaningful_blocks.py:44-74: parallel residual core inputs.
- gcnet_missing_m3/priority40_common.py:22-41: XCA/Tucker five-source features.
- gcnet_missing_m3/meaningful_blocks_set_context.py:102-122: Perceiver evidence inputs.
- gcnet_missing_m3/osram.py:1706-1717,1748-1776: joins and original Flat execution.
- gcnet_missing_m3/osram.py:1049-1054,1648-1659 and model.py:2101-2107:
  decision-correction bypass, retained inactive legacy weights, new logits.
Historical ad211c0 common wrappers and osram input join were also inspected:
these placement findings are not introduced by the recent width option.

## Existing scores, not a new ranking

MOSI, mean over eight rates / high (.5/.6/.7), percent. A dash means no
three-seed confirmation established in this bounded review, not a zero score.

| Candidate | Seed66 mean/high | 3-seed mean/high |
|---|---|---|
| Flat | 81.068 / 76.352 | 80.559 / 75.594 |
| XCA | 81.042 / 76.257 | 80.213 / 75.365 |
| Nested residual | 80.992 / 76.077 | 80.489 / 75.252 |
| Original NPS | 80.981 / 76.475 | 80.459 / 75.558 |
| D3-W256 shift filter | 80.932 / 76.200 | 79.953 / 74.992 |
| CWN | 80.821 / 76.004 | — |
| Perceiver IO | 80.818 / 76.126 | — |
| Tucker | 80.813 / 75.984 | — |
| Nested direct zero | 77.315 / 71.312 | — |
| Nested direct random | 80.177 / 74.800 | — |
| NPS + Local W96 | 80.443 / 75.648 | — |
| NPS + Local W256 | 80.343 / 75.596 | — |
| Decision correction | 79.575 / 74.781 | 79.290 / 74.164 |

Sources: osram_high_score_review_20261009/RESULT.md;
osram_readout_top3_3seed_20261005/{RESULT.md,SUMMARY.json};
osram_shift_capacity_20261002/{RESULT.md,THREE_SEED_RESULT.md};
osram_nested_direct_20261008/RESULT.md;
osram_nested_direct_random_20261008/RESULT.md;
osram_nps_local_20261009/RESULT.md; osram_nps_local_w256_20261009/RESULT.md;
osram_decision_correction_20261003/RESULT.md and its archived metrics/configs.
All paths in this paragraph are relative to experiments/.

## Earlier genuine fusion replacements: separate protocols

An independent read-only repository audit found these additional cases. They
prevent the blanket claim that no historical experiment replaced old fusion.
They are not pooled into the cfg84 shortlist above.

| Variant | Old A/S/N execution | Original task head | Evidence/status |
|---|---|---|---|
| local-cross-attn | Bypassed; new fusion has its own Skip/Norm | Retained | Five seeds66-70 completed; mean8 79.419 vs its Flat80.002, high74.518 vs75.424 |
| modality-tracks | Bypassed; new concatenation MLP/Norm | Retained | Five seeds completed; mean8 79.54 vs its baseline80.08, high75.00 vs75.52 |
| local-gated | Bypassed; own Skip/Norm | Retained | Launch evidence, no local final metrics verified in this audit |
| history-innovation | Bypassed; history/current interpolation | Retained | Cancelled archive: seeds66/67 finished,68 stopped at96; no local final scores verified |

Sources relative to experiments/: osram_local_cross_attn_20260910/results/RESULT.md;
osram_mosi_modality_tracks_20260918/{RESULT.md,results/config_seed_66.json};
osram_local_gated_20260910/LAUNCH.md;
osram_cfg84_history_innovation_20260928/{README.md,archive/gcnet_missing_m3/osram.py}.
The history-innovation files are existing untracked workspace artifacts, not
part of audited git commit 0dea9c7; they were inspected but not added to Git.
Modality-tracks uses output_dim700/value_dim32, not cfg84's1600/64. Earlier
cross-attention uses a different historical baseline; its absolute score is
not a current cfg84 comparison. Current routing evidence is in osram.py
1072-1103,1660-1675,1718-1720 and model.py2101-2107; archived interpolation
routing is in archive/gcnet_missing_m3/osram.py1170.

## What this does and does not establish

- Confirmed: the stronger shortlisted modules primarily supplement the existing
  Flat path. Their failures do not test them as independent replacement readouts.
- Confirmed: at least one completed experiment truly bypassed Flat, namely
  decision correction. But it also replaced H, changed supervision to three
  equally weighted task exits, and changed active capacity. Stored legacy Flat
  weights do not mean that Flat executed. Its new head has 433799 parameters;
  9951105 old Flat/head parameters were inactive (report's measured counts).
- Confirmed: within that new decision model, Local->LB improved mean W-F1 by
  2.4261 pp and LB->Full by 0.2368 pp (3 seeds, matched Full-selected checkpoints).
  A model can lose to Flat while its added evidence paths still contribute.
- NOT established: Flat suppresses the modules, modules were never used,
  residuals necessarily make modules ineffective, Memory lacks capacity, or
  removing Flat will improve scores. Static paths and aggregate scores cannot
  identify those causal explanations. Norm magnitude alone would not suffice.
- Zero-init bridges guarantee an initial residual of zero, not a permanently
  inactive mechanism. Common Flat adapter itself also starts with a zero last
  layer; this audit does not measure subsequent training dynamics.

## One unresolved comparison, not an authorized run

For one fixed shortlisted mechanism, supplementary versus sole contextual
readout has not been established as a clean matched comparison by the cited
Nested direct or decision-correction experiments. A future specification would
need to hold Memory, original task head, supervision and Local Skip fixed,
explicitly account for removal of A's capacity, and state initialization changes.
This would replace the contextual adapter, not delete every component called
Flat. It remains a hypothesis test, not a recommendation that replacement works.
No new module, parameter sweep or experiment is launched by this audit.
