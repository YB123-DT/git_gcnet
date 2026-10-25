# Encoder-entry paper shortlist

Literature screening only. Official papers and author repository entry points checked; not a line-by-line code audit or an authorized training launch.

## MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition (ICLR 2026)

[Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7038033f576a57a1c8469f2c6c280fcc-Abstract-Conference.html) · [Code](https://github.com/Tree-Shu-Zhao/MoRA)

Pretrained vision-language adaptation, not verified OSRAM feature-front-end improvement; different from MICCAI2024 same-name MoRA.

## Decoupling Common and Unique Representations for Multimodal Self-supervised Learning (ECCV 2024)

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04236.pdf) · [Code](https://github.com/zhu-xlab/DeCUR)

Representation objective including intra-modal anti-collapse, not two arbitrary MLP branches. Original radar/optical,RGB/elevation,RGB/depth; no OSRAM score guarantee.

## Boosting Multimodal Learning via Disentangled Gradient Learning (ICCV 2025)

[Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Boosting_Multimodal_Learning_via_Disentangled_Gradient_Learning_ICCV_2025_paper.html) · [Code](https://github.com/shicaiwei123/ICCV2025-GDL)

Training gradient-routing mechanism, not standalone inference block. Must not naively erase CHM future-task gradients; regression/missing adaptation unverified.

MoRA shares common adaptation parameters while retaining modality-specific ones. Its pretrained-adaptation setting matters; adding a low-rank linear transform to fully trainable projectors may merely reparameterize them.

DeCUR separates common/unique embedding dimensions with inter-modal correlation targets and intra-modal augmentation objectives. Do not drop its anti-collapse objective and call an arbitrary split the same mechanism. Conceptually overlaps shared/private approaches such as FactorCL.

DGL separates encoder/fusion optimization gradients. Reported MOSI experiments do not establish the present GCNet missing / CHM protocol. Not an inference Block.

SyP was not promoted as verified released code; CoMM is close to prior contrastive directions; ReMix sample recombination is not directly transferable to causal conversation sequences.

No exact shortlisted names/URLs found in searched experiment Markdown/paper-bank files; this is not comprehensive mechanism deduplication. Further paper/code/ablation inspection required, no performance claims.

