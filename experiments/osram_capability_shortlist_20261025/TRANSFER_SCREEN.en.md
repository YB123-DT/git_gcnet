# Knowledge transfer and learning schedules: second screening

Read-only research at local commit33e002e; no new training.

CMAD (ICCV2025), official repo https://github.com/YetZzzzzz/CMAD at efc812bee1549922528179cc2e94bfa066a3b2b6. Inspected Student_Model/loss.py and train_student_model_mosei.py. Feature matching includes student-teacher versus teacher-teacher similarity structure. Combination scores aggregate positive student-minus-teacher task loss gaps, normalize by the maximum, and square. The public MOSEI loop uses unit weights before begin_epoch, computes group weights at that epoch, and then holds them fixed. These weights affect both task L1 and output KD MSE. This is not continuous dynamic weighting or the project's original MSE protocol. Scalar regression temperature cancels algebraically; zero-denominator guards need testing. Teacher cost and equal-budget multi-view/uniform-KD/group-weight controls are required. Large teacher advantage does not establish that the student can infer the target. Full-paper requests returned403; complete ablation audit remains pending.

G2D (ICCV2025), https://github.com/rAIson-Lab/G2D at 27318d47de32c0b970d45d6ce6071a05b54de97f; paper https://arxiv.org/html/2506.21514v3. Inspected student.py and utils/utils.py. Separate unimodal teachers provide feature/output supervision. Public modulation uses teacher correct-class scores to set 0/1 gradient coefficients, targeting four-dimensional parameters with audio/visual top-level names. Copying this into OSRAM Linear projectors would not work. The public entry is two-modality CE classification, not a verified three-modality MOSI regression implementation. Shared historical gradients, unavailable modalities, and teacher budgets require explicit adaptation.

MCKD v1 https://arxiv.org/html/2405.07155v1 learns modality weights via meta-validation and weights cross-modal feature matching. Its original architecture also fills missing features by averaging available features; not a drop-in fit. Deprioritized due to additional optimization and architectural assumptions.

Historical correction: osram_supervised_teacher_20260914/RESULT.md reports smoke only, not full teacher/student training. Its teacher exports projector features for existing JEPA objectives. Paired history task/InfoNCE and R18 privileged noise are not equivalent to task-teacher distillation. This is targeted deduplication, not proof that no other distillation experiment exists.

Prioritize deeper CMAD component analysis, retain G2D as an alternative. No claim of OSRAM effectiveness, novelty, or fully verified matched-compute paper evidence.

