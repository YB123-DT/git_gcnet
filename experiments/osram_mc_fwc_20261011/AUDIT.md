# MC-FWC pre-implementation audit

Reference: feature/osram-uniform-forced-text, c838e14. No original CHM equations changed. User's final authorization extends scope to MOSI seeds66/67/68, 150 epochs, E0–E4, after tests and GPU smoke. No push or merge authorized.

## Inspected implementation

- model.py ObservedSetEncoder projects only observed rows; default mean fusion produces the current node. MissingM3GraphModel forwards this and modality latents to OSRAM, then the original task head.
- osram.py _project_sequence: keys depend on modality latent/current node/availability/speaker; values on modality latent. _scan reads decayed past memory before a simultaneous block_write; state is not detached. Source keys also feed current Gap residualization. Capture write-exclusive clones immediately before the actual block_write to avoid ambiguous gradient paths; these clones must be consumed by the scan.
- train_gcnet.py: original MOSI MSE / IEMOCAP CE, sample-mean task loss, clipping before optimizer.step, deterministic mask schedules and validation selection are reusable. Keep baseline entry unchanged; implement a separate training driver using these primitives.
- mixed_rate.py: cyclic batch rate and eight-rate mean validation rule can be reused. paired_views.py: dedicated augmentation RNG is a useful precedent, but neither paired masks nor InfoNCE are this method.
- MOSI loader already preserves official disjoint train/validation/test video IDs. Use one checkpoint selected on eight-rate mean validation W-F1. Historical per-rate Test-oracle baselines cannot serve as new formal performance comparisons.

## Existing experiments read (results/configuration/implementation)

- frozen_memory_audit_20261009: real-memory probe exceeds local/donor, but does not exceed/replace the original head or establish causal semantic mediation.
- write_intervention_20260909: protected writes strongly reduce association damage but do not outperform norm-matched global intervention overall; do not equate retention with task utility. Implementation is gcnet_missing_m3/evaluate_write_intervention.py, not a run.py in the report directory.
- paired_history_views_20261001 and paired_history_rho025_20261002: task-only and InfoNCE already tested; original paired config batch32, Adam1e-3, decay1e-5, clip1, latent256/output1600, eight64d heads, write_step.6, causal mean/Flat. Their test-oracle results are exploratory only.
- paired_aonly_localization_20261011: polarity bias exists in the learned no-history-evidence path; not solely created by Gap. Do not claim memory utility alone fixes this shortcut.
- future_state_20260914: differentiable post-write observer, EMA future feature target and real-batch smoke already exist; no new JEPA implementation needed.
- nested_training_gradients_20261010: finite gradients and heavy clipping already observed; zero initialization and clipping require explicit separation from a disconnected branch.
- complete150_20261011: historical150 protocol is100 epochs LR1e-3 then50 LR1e-4, full-state continuation; adopt the same schedule from scratch for the new matched validation-selected experiment.

## Mathematical and implementation decisions

L_pair=tau*softplus((ell_keep-stopgrad(ell_omit))/tau); w=sigmoid(-utility/tau). This emphasizes ineffective/harmful writes, not rewarding already useful writes. At utility0, w=.5 and loss=tau*log2. It is a counterfactual-modulated existing future task gradient, not new label information.

Compute loss per pair, mean within eligible conversation then across eligible conversations. Sample without labels, default one pair per eligible conversation, H4. Multiple pairs need separate source masking BEFORE combining gradients; masking a summed loss by a union of sources leaks credit. Omit drops one write column only and requires at least one retained source observation. Persistent trajectories cannot have s=m when m is absent throughout; report coverage rather than manufacture pairs.

Autograd first differentiates pair loss with respect to actual consumed source write nodes. Detached cotangents, masked to source conversation/modality, are VJP'd only into original key/value projector parameters. No second-order gradient and no extra encoder/query/head gradients. Normal task gradients remain intact. Sum then clip once then step once.

Default tau1.0 is in native squared-score/CE-loss units, not an optimum. Default lambda.1 is fixed a priori. E2 uses constant weight.5 to match neutral-utility scale, computes Omit for budget parity but ignores its utility. E3 has the identical softplus and normalization but full gradient. Optional shuffle preserves source/target modality, lag and regime groups; singleton groups remain unchanged and coverage is reported.

Replay full forward with the exact pre-Keep torch CPU/CUDA RNG states under no_grad, restoring post-Keep states afterward. Masks/pairs use independent deterministic generators. No interference with later training RNG. Existing layers have no running BatchNorm in cfg84; reject unsupported architectures.

Tests must demonstrate first-step writer gradients can be zero due to the existing zero-initialized adapter, and become finite/nonzero after ordinary task warm-up updates. Do not change initialization to manufacture coverage.

## Scope and risks

No new parameters, inference module, loss head, sequential A/T/V writes, completion, relation graph or test-label selection. Counterfactual utility includes joint solve coupling and subsequent evolution. Training cost is greater than E0/E1 and must be measured. Validation may reject the method; a successful smoke is not a performance claim. Formal E0–E4 outcomes, plus optional shuffled control, are required to attribute any benefit to weights and gradient direction rather than trajectory exposure.
