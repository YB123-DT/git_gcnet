# No-update audit: inherited anisotropy and loss-gradient balance

Seed66, MOSI, source implementation `ea62697`. Existing two-update Stage1 artifact only. **Zero optimizer steps; no new training, no loss/temperature changes, no Test batches.** Results are a single-seed, single-training-batch diagnostic, not a population-level claim or a causal explanation of F1.

## 1. Rank1.84 is not evidence that Stage1 collapsed a rich Teacher space

All rows use the same frozen supervised Teacher export and identical samples within each split. Effective rank is the exponential entropy of normalized **centered covariance eigenvalues** (squared singular values); do not mix it with entropy of unsquared singular values.

| Representation | Train (1,284) | Validation (229) |
|---|---:|---:|
| Teacher Text, 256d | 1.6358 | 1.6437 |
| Initial random R0(Teacher Text), 32d | 1.4574 | 1.4622 |
| Selected R(Teacher Text), 32d, after two updates | 1.8531 | 1.8410 |

The Teacher is already strongly anisotropic; random projection inherits it. The two updates **increase**, rather than decrease, effective rank relative to R0. This contradicts the specific explanation that these updates compressed an initially diverse representation into rank1.84.

It does not prove the subspace is diverse enough, nor that R creates information. A linear map can redistribute variance and increase entropy-based effective rank without increasing algebraic rank or adding sample information. Teacher256 and R32 also have different maximum ranks. The prior `collapsed` boolean is only a threshold flag; it cannot identify optimization-induced collapse.

R0 was not stored originally. It was reconstructed using the exact original seed, Teacher/data-loading and module-initialization order. The original first-batch R sentiment/prediction and Q prediction gradient norms reproduce to numerical roundoff. This is a replayed initialization, not a previously saved R0 snapshot. `DYNAMICS.json` includes the parameter hash, Teacher hash, checkpoint hash, first-batch index and gradient match.

## 2. Stage1: sentiment and covariance both exert large gradients

All values below are unweighted/pre-clip gradients on the **same R parameters and same original first training batch**. All four coefficients are1. The selected R/C/Q is evaluated on that same batch; no intermediate optimizer state was reconstructed.

| R gradient L2 | Initial | After two updates, same batch |
|---|---:|---:|
| Sentiment | 22.5791 | 12.4911 |
| Predictability | 1.09185 | 0.74127 |
| Variance | 1.79773 | 1.64975 |
| Covariance | 24.9106 | 8.18812 |

Predictability is much smaller than sentiment (20.68× initially), but covariance is initially **larger** than sentiment. Thus the stronger supported description is “sentiment/covariance dominate the gradient magnitudes, with weaker predictability,” not “only sentiment shapes R.”

| Pairwise R gradient cosine | Initial | After two updates |
|---|---:|---:|
| Sentiment vs predictability | +0.2386 | -0.2636 |
| Sentiment vs variance | -0.4156 | +0.2397 |
| Sentiment vs covariance | +0.2263 | -0.3081 |
| Predictability vs variance | -0.7261 | -0.7911 |
| Predictability vs covariance | +0.7942 | +0.8795 |
| Variance vs covariance | -0.8382 | -0.8764 |

The four terms are not mutually aligned. In particular, variance and covariance gradients oppose one another substantially; covariance also aligns with predictability on this batch. These are geometric observations, not proof that one term is helpful or harmful to final sentiment classification.

Full R/C/Q total-gradient norms are37.4677 initially and12.8117 after two updates. With the existing threshold1 clip, the hypothetical common scaling coefficients are0.02669 and0.07805. **This audit does not apply clipping.** Global norm clipping scales all components together; it does not by itself change their relative magnitudes/cosines. Adam moments/preconditioning and weight decay further mean raw gradient magnitude is not a percentage attribution of the actual parameter update.

## 3. Stage2: auxiliary strength and compatibility on actual shared coordinates

The original fresh Student and exact missing=.5 batch/dropout realization were reconstructed. Emotion/regression/InfoNCE losses reproduced as2.64912/0.126591/10.85580. No Stage2 post-update checkpoint was saved in the original smoke, so the measurements are **before its first optimizer update**, not a trained-Student audit.

Primary comparison uses parameter tensors participating in both objectives. It excludes independent MMoE/classifier heads, frozen Teacher/R, and OSRAM components used only by emotion. Both gradients are embedded in the same coordinate list; zero-gradient cosine is null, not zero.

| Shared group | Emotion L2 | 0.1 JEPA L2 | Cosine |
|---|---:|---:|---:|
| ObservedSetEncoder | 6.44215 | 2.48536 | +0.01955 |
| OSRAM common read/write parameters | 0 | 2.01090 | Undefined |
| Combined (2,179,616 parameters) | 6.44215 | 3.19698 | +0.01520 |

The combined auxiliary/main norm ratio is **49.63%**; the dot product is small positive, not strong conflict. This does not establish later compatibility or prove that a near-orthogonal auxiliary gradient helps.

The OSRAM zero emotion gradient is expected **at this initialization**: the final Flat `emotion_adapter` layer is zero-initialized. The context→emotion path has zero gradient back to memory/read-write parameters on the first step. Emotion still has nonzero gradients to OSRAM local/readout parameters, which are not part of the common JEPA support. Therefore do not conclude that OSRAM permanently receives no emotion supervision.

For completeness, whole encoder+OSRAM modules (4,601,820 parameters, including emotion-only components) give emotion9.85850, weighted JEPA3.19698, cosine+0.00993. The full parameter names and both coordinate definitions are preserved in JSON so the support is auditable.

### Contrastive vs regression on those same shared parameters

The exact total-loss coefficients are0.05 for each branch: `0.1*(0.5*reg + 0.5*NCE)`.

| Contribution | Shared gradient L2 |
|---|---:|
| 0.05 × regression | 0.006669 |
| 0.05 × InfoNCE | 3.196679 |
| Their cosine | +0.04466 |

NCE/regression norm ratio is **479.36×**. Hence the actual first-step auxiliary gradient is overwhelmingly contrastive, even though both scalar coefficients equal0.5 internally. This supports the proposed scale concern, without changing either weight or temperature.

Correction to gradient notation: the old smoke's output-gradient0.00265 and3.573 came from differentiating separate **unweighted regression and NCE losses**, respectively, not from differentiating the combined JEPA loss. Their ratio was meaningful, but the weighting must be explicit. The shared-parameter comparison here includes both0.5 and0.1.

## 4. Keep the predictability claim local and cross-modal

Stage1 learns a **task-relevant cross-modal predictable Text subspace**, where predictability is defined from current frozen Teacher Audio/Visual latents. Stage2 instead uses Student latents plus causal Base/Gap context. They are different information sets and coordinate systems. No evidence here permits claiming predictability from all Stage2-observable context, or guarantees that useful context-only directions were preserved by R.

## 5. What changes now

Only this independent audit, tests and report were added. Production model, Stage1/Stage2 losses, clipping, temperature, mask protocol and training runners are unchanged. Do not label erank1.84 as newly induced R collapse. Retain the low-diversity concern and the documented contrastive gradient imbalance; neither is yet a proven cause of Student F1 failure.

No additional training or automatic weight tuning was launched.

## Reproduction / verification

```bash
# Run from the repository with its established PyTorch environment.
python experiments/osram_text_subspace_20260915/audit_dynamics.py --output /path/to/new_audit
python -m pytest tests/test_text_subspace_dynamics.py -q
```

GPU6, original frozen MOSI features and Teacher checkpoint. Model parameter/buffer hashes unchanged and every `.grad` remainsNone. Full audit intentionally reconstructs/consumes RNG for the original dropout realization; it does not promise globally unchanged RNG. Pure rank and gradient helpers do not change parameters or accumulate gradients.

Evidence: `DYNAMICS.json`; specification: `DYNAMICS_PLAN.md`. Independent review found no blocking issue. **48 related tests passed** (one pre-existing PyG deprecation warning); Python compile and `git diff --check` passed. Single-seed diagnostics have no meaningful across-seed confidence interval; no significance claims are made.
