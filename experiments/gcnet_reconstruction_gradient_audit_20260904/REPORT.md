# GCNet modality reconstruction gradient audit

## Scope

- Datasets: IEMOCAP-4, IEMOCAP-6, CMU-MOSI, CMU-MOSEI.
- Missing rates: 0.0 through 0.7 in increments of 0.1.
- Seed: 66.
- Model: Original GCNet with the shared linear reconstruction head and the
  corrected missing-only reconstruction objective.
- Gradient sites:
  - post-graph shared hidden representation;
  - all shared backbone parameters, excluding the classifier and reconstruction
    head.

The three losses are exactly the Audio, Text, and Visual terms of GCNet's
`MaskedReconLoss`. Gradients on the complete `linear_rec.weight` tensor are not
used for conflict claims: each modality touches a disjoint output-row block, so
their global head gradients are orthogonal by construction.

At missing rate 0.0, all three reconstruction terms and gradients are exactly
zero. The empty cosine entries at that rate therefore mean *not applicable*,
not zero similarity.

## Shared-backbone gradient norm and cosine

Each row shows `Audio / Text / Visual` mean backbone gradient norms, followed by
the aggregate `A-T / A-V / T-V` cosine similarities.

### IEMOCAP-6

| rate | gradient norm A / T / V | cosine A-T / A-V / T-V |
|---:|---:|---:|
| 0.1 | .0021 / .0234 / .0170 | +.171 / -.035 / +.260 |
| 0.2 | .0043 / .0424 / .0243 | +.189 / -.175 / -.074 |
| 0.3 | .0058 / .0529 / .0437 | +.360 / -.207 / -.076 |
| 0.4 | .0078 / .0614 / .0311 | -.154 / -.120 / +.134 |
| 0.5 | .0090 / .0609 / .1041 | +.076 / +.004 / -.357 |
| 0.6 | .0102 / .0672 / .0486 | +.067 / +.015 / +.248 |
| 0.7 | .0113 / .0668 / .0610 | +.149 / +.021 / +.080 |

### IEMOCAP-4

| rate | gradient norm A / T / V | cosine A-T / A-V / T-V |
|---:|---:|---:|
| 0.1 | .0032 / .0227 / .0111 | -.351 / +.094 / +.021 |
| 0.2 | .0056 / .0228 / .0171 | +.097 / +.060 / -.020 |
| 0.3 | .0094 / .0522 / .0261 | -.361 / +.076 / +.066 |
| 0.4 | .0107 / .0453 / .0514 | -.083 / +.007 / +.260 |
| 0.5 | .0130 / .0490 / .0527 | -.060 / -.006 / +.253 |
| 0.6 | .0149 / .0511 / .0391 | +.037 / +.025 / +.053 |
| 0.7 | .0167 / .0648 / .0843 | +.023 / -.051 / -.022 |

### CMU-MOSI

| rate | gradient norm A / T / V | cosine A-T / A-V / T-V |
|---:|---:|---:|
| 0.1 | .0022 / .0162 / .0234 | -.117 / -.128 / +.663 |
| 0.2 | .0038 / .0403 / .0466 | -.108 / -.114 / +.609 |
| 0.3 | .0044 / .0590 / .0581 | -.106 / -.120 / +.663 |
| 0.4 | .0074 / .0634 / .0777 | -.094 / -.134 / +.654 |
| 0.5 | .0093 / .0710 / .0862 | -.098 / -.117 / +.628 |
| 0.6 | .0106 / .0915 / .0983 | -.099 / -.123 / +.655 |
| 0.7 | .0113 / .0881 / .1271 | -.084 / -.121 / +.659 |

### CMU-MOSEI (epoch-33 best-checkpoint snapshot)

| rate | gradient norm A / T / V | cosine A-T / A-V / T-V |
|---:|---:|---:|
| 0.1 | .0004 / .0324 / .0638 | -.161 / -.188 / +.192 |
| 0.2 | .0007 / .0625 / .0962 | -.100 / -.127 / +.188 |
| 0.3 | .0010 / .0672 / .1244 | -.179 / -.148 / +.167 |
| 0.4 | .0014 / .0982 / .1760 | -.204 / -.125 / +.131 |
| 0.5 | .0016 / .1033 / .1796 | -.194 / -.129 / +.160 |
| 0.6 | .0018 / .1292 / .2228 | -.191 / -.126 / +.132 |
| 0.7 | .0021 / .1410 / .2339 | -.187 / -.118 / +.188 |

## Findings

1. The reconstruction objective is severely scale-imbalanced. Dividing each
   squared-error sum by feature dimension does not normalize upstream feature
   scale. Text and Visual usually dominate Audio; the imbalance is especially
   large on MOSEI.
2. MOSI has the clearest directional structure. Audio reconstruction opposes
   both Text and Visual at every non-zero rate, while Text and Visual are
   strongly aligned. This is not random six-task conflict.
3. IEMOCAP conflicts are rate-dependent rather than persistent. Consequently,
   a single fixed modality weighting or generic gradient surgery is unlikely to
   transfer cleanly across datasets.
4. Direct hidden gradients are mostly near orthogonal. The stronger alignment
   and conflict emerge after multiplication by the shared GCNet Jacobian, so
   inspecting only reconstruction-head gradients would miss the relevant
   interaction.
5. MOSEI was checked at two separated best-checkpoint snapshots (epochs 13 and
   33). Both show weak Audio gradients, Audio-versus-Text/Visual opposition, and
   Text-Visual alignment; the conclusion is not an early-checkpoint accident.

## Checkpoint provenance

The legacy GCNet reproduction command saved `.npz` predictions and curves but
not model parameters. Of 303 archived `best.pt` files, only the IEMOCAP-6
Original control contained `linear_rec.*`. IEMOCAP-4, MOSI, and MOSEI therefore
required one seed-66 Original matched-control run to create auditable
checkpoints. Missing-M3 checkpoints were not substituted because they do not
contain the Original linear reconstruction head.

Raw machine-readable results and exact checkpoint paths are stored beside this
report. The diagnostic maps historical PyG `GraphConv.lin_l/lin_r` names to the
installed `lin_rel/lin_root` names without changing tensor values.
