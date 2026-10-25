# Capability shortlist: modality learning competition and complementary specialization

INTERNAL RESEARCH SHORTLIST — NOT AN IMPLEMENTATION OR PERFORMANCE RESULT

Source anchor: `65e3a66a7da88306c28e2c0929000d0563f10b3f`, branch `feature/ecerc-features`. This task changes no model and launches no training. The following mechanisms belong to one capability family, not three independent research directions.

## Candidate capability

Give the trainable A/T/V input branches relatively independent opportunities to learn complementary predictive information, rather than changing fusion only after Local/Base/Gap have been produced. Modality competition in OSRAM is a hypothesis, not an established diagnosis. This is a training capability, not a standalone inference Block.

`ObservedSetEncoder` in `gcnet_missing_m3/model.py` has independent `ModalityProjector` modules before fusion, providing explicit parameter boundaries. Inputs remain frozen pre-extracted utterance features; this would not train the original foundation encoders. CHM joint writing, queries, reads and Flat can retain their definitions.

## Alternative mechanisms: choose one, do not combine

1. **AGM (ICCV 2023):** modality-contribution-dependent gradient modulation. The [paper](https://arxiv.org/abs/2308.07686) abstract and [official repository](https://github.com/lihongcs/AGM) README were verified; the implementation was not fully audited. This is a relatively low-intrusion training comparison. Balanced gradients are not themselves a successful task outcome; AGM should not be paraphrased as always suppressing text.
2. **ReconBoost (ICML 2024):** alternating modality updates with dynamic reconciliation, consolidation and global rectification. See the [proceedings](https://proceedings.mlr.press/v235/hua24a.html), [paper](https://arxiv.org/html/2405.09321v1) and [author code](https://github.com/huacong/ReconBoost). `train_MSA.py` was read directly. It uses integer labels and classification CE, so it is not a drop-in reproduction of OSRAM sentiment MSE. Both inputs of its displayed `ga_loss` are detached: that term alone cannot deliver a parameter gradient. This is a code-path observation, not a rejection of the paper.
3. **MLA (CVPR 2024):** alternating unimodal adaptation and protection against shared-head forgetting. Its [paper](https://arxiv.org/html/2311.10707v2), including method and ablations, was read. Both advertised GitHub paths returned 404 in this session. The full method also uses uncertainty-based test-time fusion; transferring only its training mechanism must be labeled a partial mechanism adaptation. Do not prioritize implementation until the code is recovered.

The MLA ablation separates shared-head gradient modification from inference fusion; it is not merely an attention comparison. Its reported gains do not predict OSRAM gains.

## Deduplication

C20 CAGrad already balances Local/Base/Full supervised exits; that is distinct from regulating A/T/V input branches. Its seed66 advantage over its mean-gradient control was only approximately 0.014 pp. FactorCL, Fishr, LUPI, RNC, paired InfoNCE and hierarchical decision correction were actually trained; broad claims that training objectives or complementarity have never been investigated would be false. V3, TASR, Nested and Source-Role generally process already-produced reads instead.

No actual OGM/AGM or ReconBoost implementation/run was found in the searched registries, inventory and relevant code. This is a bounded search result, not an exhaustive historical nonexistence claim. Evidence: `../osram_all_block_inventory_20261023/` (README, TRAINING_INPUT_NOTES, ALL_BLOCKS, RUN_RECORDS), `../osram_core20_20261005/` (RESULT and THREE_SEED_RESULT), and `../osram_r12_3seed_20261005/RESULT.md`. Later inventory entries supersede stale pending statuses for FactorCL/Fishr.

## Proposed falsifiable comparison, not execution authorization

After selection, write an implementation specification. For a ReconBoost-inspired transfer, compare original OSRAM, matched plain alternating updates, and matched alternating updates with a complementary objective. Match model capacity, actual optimizer steps, legal GCNet missing trajectories and data exposures; record compute and RNG differences. Equal epochs alone do not establish equal budgets. Retain original emotion supervision, never pass labels into the Writer, and keep the joint Block Write. A regression version requires an explicit derivation; renaming classification KL/CE to MSE is not a faithful transfer.

Evaluate matched eight-rate and high-missing ACC/W-F1/MSE. If the complementary method beats neither plain alternation nor baseline, reject this implementation. If both alternating methods improve equally, attribute the evidence to their shared schedule, not complementary learning. Positive results remain limited-seed internal Test-oracle exploration, not proof that modality competition was solved.

## Other gaps not promoted

Causal inference in actual emotion-label space differs from historical B2 same/change supervision and latent evidence-token CRF, but this search did not establish a sufficiently compelling modern open-source transfer. Explicit cross-modal delay estimation was not found in the inventory, but aligned utterance features lack suitable fine-grained timing; lag embeddings would not establish alignment ability.

## Current recommendation and limitations

Discuss ReconBoost-like complementary specialization first, with AGM as a lower-risk alternative. Keep MLA pending code recovery. No new result, confirmed defect, or implemented Block is claimed. If an independent inference-time second Block is mandatory, this family does not meet that requirement; do not relabel an optimizer as a network module. Await user selection; no automatic experiment launch.
