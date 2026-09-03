# MOSI target-conflict diagnostic

## Question

Does one optimizer step contain different missing targets, and do their gradients interfere inside the shared MMoE experts?

## Setup

- Dataset: CMU-MOSI
- Seed: 66
- Batch size: 32 conversations
- Protocol: `train_rate_mode=all`
- Checkpoint: existing Missing-M3 all-rates seed-66 checkpoint, selected at epoch 44
- Shared parameters measured: `missing_predictor.mmoe.experts.*`
- Rates: 0.1 through 0.7; rate 0.0 has no missing target
- Diagnostic only: no optimizer update and no model training

The checkpoint used legacy PyG `GraphConv` state keys. For diagnostic loading only, `lin_l` was mapped to `lin_rel` and `lin_r` to `lin_root`; all tensors then loaded with no missing or unexpected keys.

## Target co-occurrence

The full MOSI training split contains 1,284 valid utterances in 52 conversations. Target counts at epoch 44 were:

| Missing rate | Missing A | Missing T | Missing V | Conversations containing at least two target types |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 | 0 | 0 | 0 / 52 |
| 0.1 | 129 | 143 | 129 | 50 / 52 |
| 0.2 | 266 | 240 | 241 | 52 / 52 |
| 0.3 | 374 | 358 | 354 | 52 / 52 |
| 0.4 | 500 | 458 | 471 | 52 / 52 |
| 0.5 | 580 | 612 | 587 | 52 / 52 |
| 0.6 | 660 | 692 | 668 | 52 / 52 |
| 0.7 | 719 | 787 | 738 | 52 / 52 |

The observed-pattern counts were:

| Rate | V (`001`) | T (`010`) | TV (`011`) | A (`100`) | AV (`101`) | AT (`110`) | ATV (`111`) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 | 0 | 0 | 0 | 0 | 0 | 1,284 |
| 0.1 | 16 | 13 | 100 | 12 | 115 | 104 | 924 |
| 0.2 | 41 | 46 | 179 | 44 | 155 | 151 | 668 |
| 0.3 | 88 | 78 | 208 | 100 | 170 | 176 | 464 |
| 0.4 | 153 | 166 | 181 | 133 | 172 | 172 | 307 |
| 0.5 | 208 | 216 | 156 | 220 | 184 | 151 | 149 |
| 0.6 | 273 | 278 | 109 | 271 | 148 | 119 | 86 |
| 0.7 | 340 | 305 | 74 | 356 | 91 | 77 | 41 |

At rate 0.1, 50 of 52 conversations already contained at least two different missing-target types; from rate 0.2 onward this was true for all conversations. Therefore target mixing is confirmed across every nonzero official missing rate rather than only in one selected batch.

In `train_rate_mode=all`, the same conversation batch is evaluated at all eight rates before one `optimizer.step()`. Gradients from all nonzero rates and all three target modalities therefore reach the shared experts in the same update.

## Gradient cosine evidence

Pairwise cosine values below are means over the two training batches for each rate. Negative counts show how many of the two batches produced a negative cosine.

| Rate | A–T mean / negative | A–V mean / negative | T–V mean / negative |
| ---: | ---: | ---: | ---: |
| 0.1 | -0.0027 / 1 | -0.0006 / 1 | 0.0085 / 1 |
| 0.2 | -0.0050 / 1 | 0.0236 / 1 | 0.0113 / 1 |
| 0.3 | 0.0365 / 1 | 0.0177 / 0 | 0.0361 / 0 |
| 0.4 | -0.0222 / 2 | 0.0248 / 0 | 0.0441 / 0 |
| 0.5 | 0.0194 / 1 | 0.0008 / 1 | 0.0249 / 1 |
| 0.6 | -0.0136 / 1 | 0.0073 / 1 | 0.0750 / 0 |
| 0.7 | 0.0328 / 0 | 0.0158 / 0 | 0.0616 / 0 |

Across 42 pair observations, 14 were negative. Most means were close to zero, indicating weak cooperation rather than uniformly severe opposition.

After accumulating all seven nonzero rates exactly as one all-rates update:

| Pair | Batch cosines | Mean |
| --- | --- | ---: |
| A–T | -0.0483, 0.0750 | 0.0134 |
| A–V | 0.0002, 0.0293 | 0.0148 |
| T–V | 0.0408, 0.0561 | 0.0484 |

The aggregate shared-expert gradient norms were:

- batch 1: A 0.7644, T 0.4857, V 2.8558
- batch 2: A 0.6535, T 0.8478, V 1.9756

Visual-target gradients were roughly 3–4 times larger than the other target gradients, which is stronger evidence of target domination than of consistently negative gradient conflict.

## Verdict

1. Different samples and utterances predicting A, T, and V in the same update: **confirmed**.
2. Target gradients being naturally cooperative: **not supported**; their cosine is usually near zero and is sometimes negative.
3. Uniformly severe destructive conflict: **not confirmed**.
4. Target imbalance, especially Visual-target domination: **confirmed for this checkpoint and seed**.
5. Final harm to emotion performance: **not yet causally confirmed**. The required causal test is the paired five-seed `target_private_rank=0` versus `32` experiment under the same environment, masks, seeds, and checkpoint-selection rule.

This diagnostic supports Target-Private Expert Residual as a falsifiable treatment, but it must not be described as an already proven performance improvement.
