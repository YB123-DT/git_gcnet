# GCNet Second-Graph Mechanism Experiment

Chinese version: [EXPERIMENT.zh.md](EXPERIMENT.zh.md). English mirror: [EXPERIMENT.en.md](EXPERIMENT.en.md).

## Status boundary (2026-08-25)

This document now records the completed preregistered discrimination experiment. All four mechanisms were implemented and all 24 candidate tasks finished successfully, but all four failed the advancement gate. Detailed evidence is available in [RESULTS](results/RESULTS.md), [Chinese RESULTS](results/RESULTS.zh.md), [English RESULTS](results/RESULTS.en.md), [ANALYSIS](results/ANALYSIS.md), [Chinese ANALYSIS](results/ANALYSIS.zh.md), and [English ANALYSIS](results/ANALYSIS.en.md).

| Candidate | Current state | Evidence boundary |
|---|---|---|
| GenAgg | `IMPLEMENTED; 6/6 SUCCESS; GATE FAIL` | Core implementation commit `f34405993b96dcfcc64c7867dd82af5a54415073`; GCNet integration `183369c655c200c7a96d5fed84bd0b16519728be`; training identity `dae2903a99744ec9a95a7294373a2c4713c12fd9`; runner `bad59fd25130ebc83726d44f3832e682e46cc795`. |
| Scaled Soft Medoid | `IMPLEMENTED; 6/6 SUCCESS; GATE FAIL` | Same four Phase-A implementation commits; all six canonical archives passed provenance validation. |
| SSMA Conv2 | `IMPLEMENTED; 6/6 SUCCESS; GATE FAIL` | Core commit `08aa55fb255d5e32aa9f6171246e6e2821c97c71`; two-branch/CLI/runner integration `24ea3e7bfb65621d48d935291cb233db69f54dcc`. |
| Custom RTDR | `IMPLEMENTED; 6/6 SUCCESS; GATE FAIL` | Core commit `8f375b2509016daf5395863b0220591bc8bcd3ee`; CLI/runner commit `a107f7448978f4c22f87a6b61ec45b53da312aa0`; zero added trainable parameters. |
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

## Original inheritance

The 40 existing Original archives are read-only controls:

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
  protocol_recovery_v1_biggpu/formal/original
```

They were joined to candidates by `(missing_rate, seed, fold)` and mask SHA256; they were not retrained. Commit `d515386f3207105c8207c34eca3f9743d2b80e4f` implemented the fail-closed legacy-aware validator. It allows only historically absent fields to map to the locked defaults `branch_fusion=addition`, `pre_graph_context=bilstm`, `post_graph_context=bilstm`, `second_graph_aggregation=add`, and `relation_track_routing=early`, while strictly checking source/run manifest, command, dataset, fold, features, seed, rate, parameter count, and mask hash. Validation passed for Original 40/40 and candidates 24/24 before the summary was accepted.

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

Thus all four candidates stop after their six-task discrimination subset; none is expanded to eight missing rates × five seeds. The 24 provenance-valid completed runs remain scientific negative evidence rather than infrastructure failures. See [RESULTS](results/RESULTS.md) and the detailed [ANALYSIS](results/ANALYSIS.md).

SSMA additionally requires a parameter-matched sum-plus-MLP control before any claim that gains arise from neighbor interaction rather than parameter count. RTDR must first show that its `early` path is bit-exact Original and its `full-transition` decomposition agrees with Original within forward tolerance `1e-6` and backward tolerance `1e-5`.

## Automatic result publication to GitHub

The publication target is the configured remote `github`, whose URL is [https://github.com/YB123-DT/git_gcnet](https://github.com/YB123-DT/git_gcnet). Training and result validation are complete. Publication has not yet been claimed as pushed; after final repository verification it will be completed in this same work turn without another user prompt.

The automation performs the following ordered gate:

1. Treat the completed 24/24 candidate archives and validated 40/40 Original controls as the immutable evidence set; failed GPU-4 attempts stay in diagnostics only.
2. Keep code, task-level NPZ results, logs needed for audit, run/invocation manifests, source/hash manifests, `RESULTS.md/.zh.md/.en.md`, and `ANALYSIS.md/.zh.md/.en.md` in the authoritative local workspace under `/data2/yb/paper`.
3. Do not upload datasets, extracted feature tensors, mask-bank payloads, environment credentials, device-login tokens, caches, checkpoints without a declared need, or absolute-path-only symlinks. Publish the mask hashes and provenance needed to reproduce pairing.
4. Re-run repository tests and result/provenance validation on the exact tree to be published. Record candidate status as `PASS`, scientific `FAIL`, or infrastructure `INCOMPLETE` without converting one category into another.
5. Place only verified completed versions in the organized repository layout, create a Lore-format commit whose `Tested:` trailer names the validation commands and whose `Not-tested:` trailer states any remaining gap, then push the current experiment branch with:

```bash
git push github HEAD:refs/heads/exp/second-graph-aggregators
```

6. Confirm publication by comparing the local commit with `git ls-remote github refs/heads/exp/second-graph-aggregators`. A transport failure is retried from the same local commit; training is not rerun. Promotion into the GitHub `main` completed-version layout occurs only from that verified commit, without force-pushing or rewriting unrelated completed versions.

This upload step will publish the code and real completed negative results. The document does not claim that the push has already happened; remote-SHA confirmation is the final publication evidence.
