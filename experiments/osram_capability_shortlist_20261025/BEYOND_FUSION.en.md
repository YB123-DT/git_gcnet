# Search beyond repeated fusion

Repository anchor: 6330051. Literature screening only; no training, model edits, or complete implementation audit.

Constraints: retain upstream feature extractors and CHM mathematics; do not rename existing fusion mechanisms as new capabilities.

MLA (CVPR 2024) alternates unimodal optimization, protects a shared head using gradient modification, and fuses predictions using uncertainty. This is a training/prediction paradigm, not a drop-in second block. Both repository links listed by the paper returned 404 through the web tool; implementation remains unverified. Compatibility with historical BPTT and distinctions from existing gradient and multi-view experiments require investigation.

MiDl (ICLR 2025) performs unlabeled missing-modality test-time adaptation with mutual-information and self-distillation objectives. Its official repository README is accessible, but implementation has not been audited. It overlaps the prior MEMO direction at the adaptation-paradigm level, changes evaluation requirements, and needs regression-specific adaptation. Not prioritized for immediate training.

No independent second block satisfying all constraints was confirmed in this bounded search. This does not establish impossibility. Feature-space transformations need not retrain upstream extractors, but a distinct useful capability cannot be inferred from their placement alone.

Sources:
- https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multimodal_Representation_Learning_by_Alternating_Unimodal_Adaptation_CVPR_2024_paper.html
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html
- https://github.com/meryusha/midl_tta

Deduplication was limited to the existing block catalog and targeted code searches, not a comprehensive equivalence audit.

