# Fixed-feature OSRAM: literature screening
Date: 2026-10-11. Repository reference: 8ef92c7. Literature only; no model changes or new runs.

The user rejected changing raw feature extractors. Avoid repackaging earlier five-slot graphs, scalar gates, or residual operators. All transfers below are untested hypotheses, not demonstrated MOSI improvements or novelty claims.

## TabM: conditionally retain for stronger Flat learning
[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf), [reference implementation](https://github.com/yandex-research/tabm/blob/main/tabm_reference.py), [training code](https://github.com/yandex-research/tabm/blob/main/paper/bin/model.py).
Shared matrices with member-specific input/output scales and biases implement s_j * W(r_j * x)+b_j. Train the mean of member losses, not the loss of the mean prediction. Average regression predictions or classification probabilities at inference.
Proposed adaptation: one OSRAM scan, diversified members before the first Flat fusion layer. Readout compute and activations grow despite modest extra parameters.
The paper did not investigate normalization backbones (Appendix A.4); OSRAM LN/Skip transfer is not a validated reproduction.
Previously listed but not counted in docs/osram_survey80_20261004/representation.json because individual-head supervision was excluded. Not a newly discovered or previously completed run. Changing that constraint needs an explicit decision.
Audit: paper methods/limitations and official model/training implementation.

## TabR: conditionally retain for training-instance evidence
[Paper v2](https://arxiv.org/pdf/2307.14338), [pinned official code](https://github.com/yandex-research/tabular-dl-tabr/blob/17baa9082506f8e7a0f8d11bb1e08212926a1507/bin/tabr.py#L235-L312).
Retrieve training keys with negative squared-distance weights; values combine training-label embeddings with transformed query-neighbor differences. The method still injects a residual.
Proposed adaptation: query other training conversations using lawful L/B/G/a, without writing retrieved evidence to persistent OSRAM memory. This changes the evidence source rather than only transforming existing slots.
Require conversation exclusion beyond the original self-exclusion, fold-specific training banks, and no test labels. Missing-support mismatch, wrong neighbors, stale bank representations, and re-encoding/storage costs are unresolved. It does not recover target-specific missing Text.
Audit: arXiv v2 and full official core implementation. Final conference PDF access was blocked; not claimed audited. No repository name match does not rule out equivalent mechanisms.

## RealMLP: training reference only
[Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf), [author repository](https://github.com/dholzmueller/pytabkit).
A combined preprocessing, parametrization, initialization, and optimization recipe rather than a new local block. Learned OSRAM features are not static tabular inputs. Paper and repository entry inspected; detailed implementation audit incomplete.

## Hyper-Connections: reject for current use
[Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1e8ff97057e0c3f6bcb7018c78d4df1-Paper-Conference.pdf).
Expanded residual streams and learned width/depth mixing target propagation through deep networks. Main text and author pseudocode in Appendix J inspected. No sufficient reason to deepen a shallow Flat merely to accommodate this mechanism. Third-party libraries are not author code.

## Decision
TabM is an engineering route to strengthen Flat; TabR is a distinct evidence-source route with material deployment changes. Neither is authorized for implementation or training this turn. Existing runs remain untouched.
