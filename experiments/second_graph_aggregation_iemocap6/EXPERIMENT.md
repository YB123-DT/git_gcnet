# GCNet Second-Graph Mechanism Experiment

Chinese version: [EXPERIMENT.zh.md](EXPERIMENT.zh.md). English mirror: [EXPERIMENT.en.md](EXPERIMENT.en.md).

## Status boundary (2026-08-25)

This document records four evidence stages without rewriting their chronology. First, all four mechanisms completed the preregistered six-task discrimination subset and all four failed its advancement gate. Second, a user-directed post-gate RTDR extension satisfied its separate 15-pair extension criterion. Third, the completed 40-pair RTDR full-rate audit did **not** satisfy its separately predefined `stable_positive` criterion. Fourth, all four mechanisms were placed on a uniform minimum grid of three missing rates and five seeds. Original was never retrained. Initial evidence remains in [RESULTS](results/RESULTS.md) and [ANALYSIS](results/ANALYSIS.md); follow-ups are recorded in [uniform three-rate RESULTS](results/uniform_three_rate/RESULTS.md), [RTDR extension RESULTS](results/rtdr_extension/RESULTS.md), and [RTDR full RESULTS](results/rtdr_full/RESULTS.md).

| Candidate | Current state | Evidence boundary |
|---|---|---|
| GenAgg | `INITIAL GATE FAIL; UNIFORM false` | Uniform 15-pair delta `-0.204847963`, 0/3 positive rates, 0/5 positive seed macros, with collapse. |
| Scaled Soft Medoid | `INITIAL GATE FAIL; UNIFORM false` | Uniform 15-pair delta `+0.004706753`, 2/3 positive rates and 4/5 positive seed macros; missing `0.7` delta `-0.002089281`. |
| SSMA Conv2 | `INITIAL GATE FAIL; UNIFORM false` | Uniform 15-pair delta `-0.001153174`, 1/3 positive rates and 2/5 positive seed macros. |
| Custom RTDR | `INITIAL GATE FAIL; UNIFORM true; FULL stable_positive=false` | Uniform 15-pair delta `+0.008510981`, 3/3 positive rates and 3/5 positive seed macros; the broader 40-pair audit remained negative overall at `-0.002810103`. |
| Ego–Neighbor Separation | `REJECTED BEFORE TRAINING` | Algebraically redundant with GCNet's existing neighbor and root transforms. |
| Centered Clipping | `PARKED BEFORE TRAINING` | A source-faithful persistent center, iteration count, and threshold are not grounded for shuffled conversation nodes. |

The canonical design records are [the second-layer aggregation design](../../docs/superpowers/specs/2026-08-25-second-graph-aggregators-design.md) and [the additional-candidate decision](../../docs/superpowers/specs/2026-08-25-additional-graph-candidates-design.md). The stepwise implementation record is [the execution plan](../../docs/superpowers/plans/2026-08-25-second-graph-aggregators.md).

## Question and attribution boundary

Official GCNet uses a first-layer relation-aware `RGCNConv` followed by a second-layer `GraphConv`. The locked second layer aggregates incoming neighbors by an unweighted sum before its neighbor transform. This experiment asks two separable questions:

1. Does replacing the second-layer sum with a source-grounded nonlinear or robust set aggregator improve missing-modality emotion recognition?
2. Does suppressing off-diagonal two-hop relation transitions improve propagation without adding trainable parameters?

Only one mechanism changes in each arm. No arm is allowed to add missing-pattern conditioning, edge attention, learned adjacency, extra graph layers, contrastive objectives, uncertainty heads, or a new reconstruction objective.

## Traceable mechanism sources

- **GenAgg:** Ryan Kortvelesy, Steven Morad, and Amanda Prorok, “Generalised f-Mean Aggregation for Graph Neural Networks,” NeurIPS 2023. [Paper](https://arxiv.org/abs/2306.13826); [paper-era official implementation at commit `3c95c10`](https://github.com/Acciorocketships/generalised-aggregation/blob/3c95c10afac4bda77afc30e80a7481c7e537fca1/genagg/genagg.py).
- **Scaled Soft Medoid:** Simon Geisler, Daniel Zügner, and Stephan Günnemann, “Reliable Graph Neural Networks via Robust Aggregation,” NeurIPS 2020. [Paper](https://arxiv.org/abs/2010.15651); [official implementation at commit `4f94140`](https://github.com/sigeisler/reliable_gnn_via_robust_aggregation/blob/4f94140afb7fd2ef5bf77f45a5efc7b2d6eb2a09/rgnn/means.py).
- **SSMA:** Mitchell Keren Taraday, Almog David, and Chaim Baskin, “Sequential Signal Mixing Aggregation for Message Passing Graph Neural Networks,” NeurIPS 2024. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2409.19414); [official code inspected at commit `9d128c9`](https://github.com/AlmogDavid/SSMA/tree/9d128c902acf47343b5baf5150a78dfb6a64fb3e). The source repository README states MIT, but a root LICENSE file and GitHub license metadata were not found during screening, so the GCNet port is independently derived from the paper equations.
- **RTDR naming boundary:** the inspected “Multi-Relation Message Passing for Multi-Label Text Classification” method ([arXiv](https://arxiv.org/abs/2202.04844)) sums relation messages inside each layer and therefore does not justify delayed relation fusion. The retained **Relation-Track Diagonal Routing (RTDR)** arm is explicitly a custom hypothesis, not MrMP/MMP transfer.
- **Rejected controls:** H2GCN motivates ego/neighborhood separation ([NeurIPS 2020 paper](https://arxiv.org/abs/2006.11468)), while Centered Clipping comes from Byzantine-robust distributed optimization ([ICML 2021 paper](https://arxiv.org/abs/2012.10333)). Their rejection reasoning is recorded in the additional-candidate design rather than hidden after observing metrics.

The literature search supports a cross-domain-transfer framing for GenAgg, Soft Medoid, and SSMA; it is not an absolute first-use claim in multimodal sentiment analysis or emotion recognition.

## Exact GCNet adaptations

| Arm | Exact insertion | Mechanism and reason | Trainable-parameter delta |
|---|---|---|---:|
| Original/add | Both `GraphNetwork.conv2` instances | PyG `GraphConv`: neighbor add, then `lin_l`; separate `lin_r(x_i)` and bias. | 0 |
| GenAgg | Replace only both `conv2` aggregation primitives | Learn cardinality dependence, centralization, and nonlinear scalar transformation before the existing `lin_l`. Natural missingness changes message distributions that a fixed sum cannot adapt to. The source inverse-consistency objective is added exactly once during training. | +59/branch, +118 total |
| Scaled Soft Medoid | Replace only both `conv2` aggregation primitives | Compute a degree-scaled soft medoid over bias-free transformed messages, then add the original bias and root path once. This tests resistance to an isolated message-space outlier without assuming a learnable reliability gate. | +0 |
| SSMA Conv2 | Replace only both `conv2` aggregation primitives | Mix raw incoming neighbor states in the Fourier-domain polynomial signal, compress `2976 -> 100`, then apply the existing `lin_l`. Unlike sum, it can express cross-neighbor products before compression. Explicit self edges remain ordinary inputs. | +297,700/branch, +595,400 total |
| Custom RTDR | Replace the two-layer graph core's routing only | Preserve all existing weights and compare all relation transitions with a diagonal `q=r` two-hop transition mask. This tests relation-transition routing, not “late fusion.” | +0, verified |

The inherited Original selected path has 34,140,166 trainable parameters. The corresponding totals are 34,140,284 for GenAgg, 34,140,166 for Soft Medoid and RTDR, and 34,735,566 for SSMA. RTDR's untouched early core matched Original with maximum absolute error 0; its full-transition decomposition matched with maximum absolute error `5.96e-8` in the official Torch 1.8 check.

## What remains unchanged

All arms retain IEMOCAPSix, fold 5, the feature extractors and dimensions, immutable natural-mask bank, missing-rate generation, graph windows `2/2`, graph topology, explicit self edges, temporal and speaker relation definitions, first-layer Original RGCN for the three aggregation arms, pre/post-graph BiLSTMs, temporal/speaker addition, branch attention, classifier, reconstruction, losses except GenAgg's source-required inverse term, optimizer, scheduler, 100 epochs, and metric. SSMA does not introduce attention or sampling. RTDR may alter only the preregistered relation-transition mask.

## Completed SSMA compatibility gate

The one-time compatibility check used source commit `08aa55fb255d5e32aa9f6171246e6e2821c97c71` on biggpu with the formal interpreter and packages:

```text
Python 3.8.20
Torch 1.8.0
PyG 2.0.1
FP32 CPU forward/backward: finite
FP32 GPU forward/backward: finite
SSMA extra parameters: 297700 per branch
64-node synthetic GPU peak allocated: 56989696 bytes
64-node synthetic GPU peak reserved: 92274688 bytes
```

Torch 1.8 emitted a complex-to-real warning in the synthetic check. It did not produce a non-finite output or gradient. This gate proves runtime compatibility and bounded synthetic memory only; it is not an epoch smoke, a dataset run, or evidence of accuracy. The implementation deliberately avoids `torch.polar`, whose complex output did not provide the required Torch 1.8 automatic differentiation path in the earlier compatibility investigation.

## Locked task grid

All candidate tasks use `stage=formal`, because stage is part of the immutable artifact path and command identity. A `gate` artifact cannot later be relabeled or resumed as `formal`.

| Phase | Arms | Missing rates | Seeds | Fold | New tasks | Scheduling |
|---|---|---|---|---:|---:|---|
| A | `genagg`, `soft_medoid` | `0.0`, `0.7` | `66`, `67`, `68` | 5 | 12/12 success | Initially GPUs 4–7, 3 jobs/GPU; canonical recovery on GPUs 5–7 |
| B | `ssma`, `rtdr` | `0.0`, `0.7` | `66`, `67`, `68` | 5 | 12/12 success | GPUs 5–7, 3 jobs/GPU with automatic queue continuation |
| Total | four candidates | two rates | three seeds | 5 | 24 | two bounded 12-job waves |

Phase A initially assigned three jobs to each of GPUs 4–7. Formal-training processes on GPU 4 repeatedly exited with code `-9`; those failed attempts are retained only as diagnostics and never enter the canonical comparison. The affected three task keys were rerun on GPUs 5, 6, and 7 and succeeded. Phase B used GPUs 5–7, three concurrent jobs per GPU, with the remaining queue starting automatically as slots became free; it completed 12/12. Original was never scheduled in either wave.

### Post-gate RTDR stability audit

The original gate above remains unchanged. After seeing its result, the user explicitly requested additional RTDR stability evidence. The 15-pair extension covered missing rates `{0.0,0.5,0.7}` and seeds `{66,67,68,69,70}`; it reused the six existing RTDR tasks and required nine new trainings. Its separately locked extension rule passed with overall delta `+0.008510981`, positive means at all three rates, and 3/5 positive seed macros. The subsequent full audit covered all eight rates `{0.0,...,0.7}` and five seeds. It reused those 15 tasks and required 25 additional trainings, yielding 40/40 provenance-valid RTDR archives.

The predefined full-audit flag was

```text
stable_positive = overall macro delta > 0
                  and at least 6/8 rate means > 0
                  and at least 3/5 seed macros > 0
                  and all runs finite and non-collapsed
```

Its observed value was `false`: the overall paired F1 delta was `-0.002810103`, only 3/8 rate means were positive, and 3/5 seed macros were positive. This is a post-gate stability audit, not a retroactive advancement PASS.

### Uniform three-rate, five-seed evidence layer

The uniform layer covers rates `{0.0,0.5,0.7}` and seeds `{66,67,68,69,70}` for every arm. It required 27 new trainings: nine each for GenAgg, Soft Medoid, and SSMA; RTDR's 15 cells were reused. The formal invocation used GPUs 1–7. GPU 4 produced three code-`-9` attempts, which were moved to diagnostics; the same locked task identities then completed on GPUs 1–3. Failed attempts contribute no metric. Original was not retrained.

`uniform_stable` requires an overall positive macro delta, 3/3 positive rate means, at least 3/5 positive seed macros, and finite non-collapsed runs. GenAgg was `false` (`-0.204847963`, 0/3 rates, 0/5 seeds, collapse); Soft Medoid was `false` (`+0.004706753`, 2/3 rates, 4/5 seeds, non-collapsed; rate `0.7` was `-0.002089281`); SSMA was `false` (`-0.001153174`, 1/3 rates, 2/5 seeds); RTDR was `true` (`+0.008510981`, 3/3 rates, 3/5 seeds). This bounded RTDR descriptor does not supersede full-rate `stable_positive=false`.

The final evidence set contains 85 unique candidate archives: GenAgg 15 + Soft Medoid 15 + SSMA 15 + RTDR 40. Reused cells are counted once.

## Original inheritance

The 40 existing Original archives are read-only controls:

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
  protocol_recovery_v1_biggpu/formal/original
```

They were joined to candidates by `(missing_rate, seed, fold)` and mask SHA256; they were not retrained. Commit `d515386f3207105c8207c34eca3f9743d2b80e4f` implemented the fail-closed legacy-aware validator. It allows only historically absent fields to map to the locked defaults `branch_fusion=addition`, `pre_graph_context=bilstm`, `post_graph_context=bilstm`, `second_graph_aggregation=add`, and `relation_track_routing=early`, while strictly checking source/run manifest, command, dataset, fold, features, seed, rate, parameter count, and mask hash. Original training count remained zero throughout; the final unique candidate archive count is 85.

## Efficiency and failure protocol

1. Do not run epoch-level smoke jobs. One synthetic forward/backward per candidate in the official environment is the compatibility gate.
2. Use one clean remote Git clone made from the verified commit, not a source tree without `.git`. Compare local and remote HEAD plus a source SHA256 manifest.
3. Launch bounded parallel waves without `original`. The actual Phase-A and Phase-B GPU allocation and the GPU-4 diagnostic recovery are recorded above; completed canonical task keys were not rerun.
4. Reuse the existing stage-aware runner, locks, immutable run/invocation manifests, task keys, and resume behavior. A completed valid task is never rerun.
5. A task is complete only with return code 0, all 100 epoch records, exactly one readable NPZ, finite metrics, and matching command/mask/source provenance. Diagnose only the failing task and resume the same task key.
6. Run the full local regression suite once after the code surface is stable, not after every small selector edit.
7. Synchronize remote results back once per completed phase and validate before summarization. Do not read a half-written manifest and do not package an infrastructure-incomplete directory as a scientific result.

## Decision rule

For each candidate, every valid task was paired with inherited Original. Advancement required both rate-level paired mean F1 deltas to be positive, the seed-macro paired delta to be positive, at least two of three seed-macro deltas to be positive, and no non-finite or collapsed run.

| Candidate | Seed-macro F1 delta | Gate result | Decisive evidence |
|---|---:|---|---|
| GenAgg | -0.187831724 | FAIL | Collapse detected; macro effect strongly negative |
| Soft Medoid | -0.004304363 | FAIL | Macro effect negative |
| SSMA | -0.007173215 | FAIL | Macro effect negative |
| RTDR | +0.002541466 | FAIL | Only 1/3 seed macros positive; required at least 2/3 |

Under the original preregistered decision, all four candidates stopped after their six-task discrimination subset and none qualified for expansion. That historical gate result remains scientific negative evidence rather than an infrastructure failure. The later user-directed RTDR extension and full audit are reported separately and do not change the initial FAIL. The full audit also failed its own `stable_positive` description because its overall delta was negative and only 3/8 rate means were positive.

SSMA additionally requires a parameter-matched sum-plus-MLP control before any claim that gains arise from neighbor interaction rather than parameter count. RTDR must first show that its `early` path is bit-exact Original and its `full-transition` decomposition agrees with Original within forward tolerance `1e-6` and backward tolerance `1e-5`.

## Automatic result publication to GitHub

The earlier 58-candidate evidence snapshot was published to [https://github.com/YB123-DT/git_gcnet](https://github.com/YB123-DT/git_gcnet) as commit `97370fd49cb130bc10c620f1293ebff00985b729`; `git ls-remote` returned the same SHA. That attestation predates the 27 uniform-layer trainings and does not establish publication of the current 85-archive set.

The automation performs the following ordered gate:

1. Treat the published 58/58 candidate snapshot as an earlier evidence layer; the current uniform layer contains 85 unique candidate archives, failed GPU-4 attempts stay in diagnostics only, reused tasks are counted once, and Original training count is zero.
2. Keep code, task-level NPZ results, logs needed for audit, run/invocation manifests, source/hash manifests, the initial `RESULTS.md/.zh.md/.en.md` and `ANALYSIS.md/.zh.md/.en.md`, and the RTDR extension/full summaries in the authoritative local workspace under `/data2/yb/paper`.
3. Do not upload datasets, extracted feature tensors, mask-bank payloads, environment credentials, device-login tokens, caches, checkpoints without a declared need, or absolute-path-only symlinks. Publish the mask hashes and provenance needed to reproduce pairing.
4. Re-run repository tests and result/provenance validation on the exact tree to be published. Record candidate status as `PASS`, scientific `FAIL`, or infrastructure `INCOMPLETE` without converting one category into another.
5. Place only verified completed versions in the organized repository layout, create a Lore-format commit whose `Tested:` trailer names the validation commands and whose `Not-tested:` trailer states any remaining gap, then push the current experiment branch with:

```bash
git push github HEAD:refs/heads/exp/second-graph-aggregators
```

6. Confirm publication by comparing the local commit with `git ls-remote github refs/heads/exp/second-graph-aggregators`. A transport failure is retried from the same local commit; training is not rerun. Promotion into the GitHub `main` completed-version layout occurs only from that verified commit, without force-pushing or rewriting unrelated completed versions.

The earlier verified commit contains the first three evidence stages only. The fourth, uniform 85-archive layer passed 360/360 checksums and 267 tests (one expected skip), was published as evidence commit `f72cd776d5260644a84005241e55639b994bb1dc`, and `git ls-remote` returned the same SHA.
