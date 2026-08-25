# GCNet Second-Graph Mechanism Experiment

Chinese version: [EXPERIMENT.zh.md](EXPERIMENT.zh.md). English mirror: [EXPERIMENT.en.md](EXPERIMENT.en.md).

## Status boundary (2026-08-25)

This document is the preregistered execution and publication record, not a results claim. `IMPLEMENTED` means that code and deterministic tests exist; it does not mean that IEMOCAPSix performance improved.

| Candidate | Current state | Evidence boundary |
|---|---|---|
| GenAgg | `IMPLEMENTED; FORMAL TRAINING PENDING` | Core implementation commit `f34405993b96dcfcc64c7867dd82af5a54415073`; GCNet integration `183369c655c200c7a96d5fed84bd0b16519728be`; training identity `dae2903a99744ec9a95a7294373a2c4713c12fd9`; runner `bad59fd25130ebc83726d44f3832e682e46cc795`. |
| Scaled Soft Medoid | `IMPLEMENTED; FORMAL TRAINING PENDING` | Same four Phase-A commits. No IEMOCAPSix effect is claimed yet. |
| SSMA Conv2 | `IMPLEMENTED; OFFICIAL-ENVIRONMENT GATE PASSED; FORMAL TRAINING PENDING` | Core commit `08aa55fb255d5e32aa9f6171246e6e2821c97c71`; two-branch/CLI/runner integration `24ea3e7bfb65621d48d935291cb233db69f54dcc`. |
| Custom RTDR | `PENDING IMPLEMENTATION AND TESTS` | The design is fixed, but no code, compatibility result, parameter-count result, or training result may be inferred from this document. |
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
| Custom RTDR | Planned replacement of the two-layer graph core's routing only | Preserve all existing weights and compare all relation transitions with a diagonal `q=r` two-hop transition mask. This tests relation-transition routing, not “late fusion.” | Expected +0; `PENDING VERIFICATION` |

The inherited Original selected path has 34,140,166 trainable parameters. Therefore the locked expected totals are 34,140,284 for GenAgg, 34,140,166 for Soft Medoid, and 34,735,566 for SSMA. RTDR's equality to the Original parameter count remains a required test, not a completed fact.

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
| A | `genagg`, `soft_medoid` | `0.0`, `0.7` | `66`, `67`, `68` | 5 | 12 | GPUs 0–3, 3 workers/GPU |
| B | `ssma`, `rtdr` | `0.0`, `0.7` | `66`, `67`, `68` | 5 | 12 | GPUs 0–3, 3 workers/GPU |
| Total | four candidates | two rates | three seeds | 5 | 24 | two bounded 12-job waves |

Phase A is code-ready but its formal 12-task result set is `PENDING`. Phase B cannot launch until RTDR implementation, Original/full-transition equivalence tests, CLI identity, and runner mapping are complete. SSMA alone is ready at the implementation and environment-gate levels. Original is never placed in either candidate wave.

## Original inheritance

The 40 existing Original archives are read-only controls:

```text
/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/
  protocol_recovery_v1_biggpu/formal/original
```

They will be joined to candidates by `(missing_rate, seed, fold)` and mask SHA256; they are not retrained. Their historical run manifest records source HEAD `d64fa9b6003d9a37fef5f135ce194fd206baac2a`, Original selected-path parameter count 34,140,166, the locked feature tree, and the fixed mask bank. Before any candidate launch or comparison, a dedicated legacy-aware Original validator **must be implemented, tested, and pass**. Because these archives predate several newer provenance fields, that future validator may map only absent fields to the locked defaults `branch_fusion=addition`, `pre_graph_context=bilstm`, `post_graph_context=bilstm`, and `second_graph_aggregation=add`; it must reject drift in source/run manifest, command, dataset, fold, features, seed, rate, parameter count, and mask hash. Candidate archives must pass the current strict validation path. This paragraph specifies a pending gate and does not claim that the legacy-aware validator already exists.

## Efficiency and failure protocol

1. Do not run epoch-level smoke jobs. One synthetic forward/backward per candidate in the official environment is the compatibility gate.
2. Use one clean remote Git clone made from the verified commit, not a source tree without `.git`. Compare local and remote HEAD plus a source SHA256 manifest.
3. Launch one 12-job phase with `--parallel-arms --gpus 0 1 2 3 --workers-per-gpu 3`. Do not launch twelve serial wrappers and do not include `original`.
4. Reuse the existing stage-aware runner, locks, immutable run/invocation manifests, task keys, and resume behavior. A completed valid task is never rerun.
5. A task is complete only with return code 0, all 100 epoch records, exactly one readable NPZ, finite metrics, and matching command/mask/source provenance. Diagnose only the failing task and resume the same task key.
6. Run the full local regression suite once after the code surface is stable, not after every small selector edit.
7. Synchronize remote results back once per completed phase and validate before summarization. Do not read a half-written manifest and do not package an infrastructure-incomplete directory as a scientific result.

## Decision rule

For each candidate, pair every valid task with inherited Original. Advancement requires both rate-level paired mean F1 deltas to be positive, the seed-macro paired delta to be positive, at least two of three seed-macro deltas to be positive, and no non-finite or collapsed run. An advancing candidate reuses its six first-wave artifacts and fills the remaining rates/seeds; a rejected candidate stops. A scientifically negative but provenance-complete result is retained and published as negative evidence. An incomplete or invalid run is not called FAIL; it remains `INCOMPLETE` until repaired or explicitly archived as such.

SSMA additionally requires a parameter-matched sum-plus-MLP control before any claim that gains arise from neighbor interaction rather than parameter count. RTDR must first show that its `early` path is bit-exact Original and its `full-transition` decomposition agrees with Original within forward tolerance `1e-6` and backward tolerance `1e-5`.

## Automatic result publication to GitHub

The publication target is the configured remote `github`, whose URL is [https://github.com/YB123-DT/git_gcnet](https://github.com/YB123-DT/git_gcnet). Publication is an automatic terminal step after training; it must not require another user prompt.

The automation performs the following ordered gate:

1. Wait for both 12-job waves to reach terminal state. Validate every completed archive and the inherited Original pairing; retain valid negative results, but exclude half-written, corrupted, or provenance-mismatched artifacts from completed-version folders.
2. Synchronize code, task-level NPZ results, logs needed for audit, run/invocation manifests, source/hash manifests, and generated `RESULTS.md`, `RESULTS.zh.md`, and `RESULTS.en.md` back to the authoritative local workspace under `/data2/yb/paper`.
3. Do not upload datasets, extracted feature tensors, mask-bank payloads, environment credentials, device-login tokens, caches, checkpoints without a declared need, or absolute-path-only symlinks. Publish the mask hashes and provenance needed to reproduce pairing.
4. Re-run repository tests and result/provenance validation on the exact tree to be published. Record candidate status as `PASS`, scientific `FAIL`, or infrastructure `INCOMPLETE` without converting one category into another.
5. Place only verified completed versions in the organized repository layout, create a Lore-format commit whose `Tested:` trailer names the validation commands and whose `Not-tested:` trailer states any remaining gap, then push the current experiment branch with:

```bash
git push github HEAD:refs/heads/exp/second-graph-aggregators
```

6. Confirm publication by comparing the local commit with `git ls-remote github refs/heads/exp/second-graph-aggregators`. A transport failure is retried from the same local commit; training is not rerun. Promotion into the GitHub `main` completed-version layout occurs only from that verified commit, without force-pushing or rewriting unrelated completed versions.

This upload step publishes code and real completed results even when the scientific verdict is negative. RTDR and formal metric tables remain `PENDING` in this document until their artifacts exist and pass the gates above.
