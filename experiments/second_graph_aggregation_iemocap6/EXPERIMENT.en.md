# GCNet Second-Graph Experiment — English Record

Canonical detailed protocol: [EXPERIMENT.md](EXPERIMENT.md). Chinese record: [EXPERIMENT.zh.md](EXPERIMENT.zh.md).

## Scope and current status

This experiment changed GCNet's graph mechanism while inheriting the 40 existing IEMOCAPSix fold-5 Original archives. Original was never retrained. The chronology is preserved: all four candidates first failed the preregistered six-task advancement gate; a later user-directed 15-pair RTDR extension passed only its bounded extension criterion; the completed 40-pair RTDR audit then produced `stable_positive=false`. See the initial [RESULTS](results/RESULTS.en.md) and [ANALYSIS](results/ANALYSIS.en.md), the [RTDR extension](results/rtdr_extension/RESULTS.en.md), and the [RTDR full audit](results/rtdr_full/RESULTS.en.md).

| Candidate | Insertion | Parameters | Status on 2026-08-25 |
|---|---|---:|---|
| GenAgg | Both second-layer `GraphConv` aggregators | +118 total | Implemented; 6/6 success; gate FAIL (`-0.187831724`, collapse) |
| Scaled Soft Medoid | Both second-layer `GraphConv` aggregators | +0 | Implemented; 6/6 success; gate FAIL (`-0.004304363`) |
| SSMA | Both second-layer `GraphConv` aggregators | +595,400 total | Implemented; 6/6 success; gate FAIL (`-0.007173215`) |
| Custom RTDR | Two-hop relation-transition routing | +0 verified | Initial 6/6 gate FAIL (`+0.002541466`, 1/3 positive seeds); 15/15 extension criterion PASS; 40/40 full audit `stable_positive=false` (overall `-0.002810103`, 3/8 positive rates, 3/5 positive seeds) |

The reason for GenAgg was to learn cardinality, centering, and nonlinear set behavior that fixed sum cannot represent. Soft Medoid tested robustness to an isolated message-space outlier. SSMA explicitly modeled cross-neighbor products before compression. RTDR tested deletion of off-diagonal two-hop relation transitions; it must not be described as MrMP/MMP transfer because the inspected [MrMP paper](https://arxiv.org/abs/2202.04844) mixes relations within each layer. RTDR core commit `8f375b2509016daf5395863b0220591bc8bcd3ee` and CLI/runner commit `a107f7448978f4c22f87a6b61ec45b53da312aa0` added zero parameters; the official core checks measured errors 0 for the untouched early path and `5.96e-8` for the full-transition decomposition.

Primary sources are [GenAgg](https://arxiv.org/abs/2306.13826), [Soft Medoid](https://arxiv.org/abs/2010.15651), and [SSMA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html). The complete source/version trace and exact formulas are in [the canonical record](EXPERIMENT.md).

## SSMA official-environment evidence

Commit `08aa55fb255d5e32aa9f6171246e6e2821c97c71` was checked on biggpu with Python 3.8.20, Torch 1.8.0, and PyG 2.0.1. CPU and GPU FP32 forward/backward outputs and gradients were finite. The measured extra parameter count was 297,700 per branch. A synthetic 64-node GPU check peaked at 56,989,696 allocated bytes and 92,274,688 reserved bytes. Torch emitted a complex-to-real warning, but gradients remained finite. This is compatibility evidence only, not training or accuracy evidence.

## Two-phase locked grid

- Phase A: `{genagg, soft_medoid}` × missing `{0.0, 0.7}` × seeds `{66,67,68}` = 12 jobs.
- Phase B: `{ssma, rtdr}` × the same rates and seeds = 12 jobs.
- Both phases used `stage=formal`, fold 5, and completed 24/24 new candidate jobs with zero Original jobs.

Phase A initially used GPUs 4–7 with three jobs each. Formal-training processes on GPU 4 repeatedly exited with code `-9`; those attempts remain diagnostic-only, and the three canonical task keys were rerun successfully on GPUs 5–7. Phase B used GPUs 5–7 with three concurrent jobs per GPU and automatic queue continuation, finishing 12/12.

## Post-gate RTDR stability audit

The initial gate remains unchanged. After its result was known, the user explicitly requested additional RTDR stability evidence. The 15-pair extension covered rates `{0.0,0.5,0.7}` and five seeds, reused the six initial tasks, and required nine new trainings. Its separate extension criterion passed with overall delta `+0.008510981`, all three rate means positive, and 3/5 positive seed macros. The subsequent eight-rate, five-seed full audit reused those 15 tasks and required 25 new trainings, producing 40/40 provenance-valid RTDR archives.

The full audit predefined `stable_positive` as an overall positive macro delta, at least 6/8 positive rate means, at least 3/5 positive seed macros, and finite, non-collapsed runs throughout. Its observed value was `false`: overall delta `-0.002810103`, 3/8 positive rate means, and 3/5 positive seed macros. It is not a retroactive advancement PASS. The final evidence set contains 58 unique candidate archives: six each for GenAgg, Soft Medoid, and SSMA, plus 40 for RTDR.

The 40 read-only Original NPZ files under `/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/protocol_recovery_v1_biggpu/formal/original` were joined by rate, seed, fold, and mask SHA256. Commit `d515386f3207105c8207c34eca3f9743d2b80e4f` implemented the fail-closed legacy-aware validator. Its bounded historical defaults are `branch_fusion=addition`, `pre_graph_context=bilstm`, `post_graph_context=bilstm`, `second_graph_aggregation=add`, and `relation_track_routing=early`; Original 40/40 and the initial candidates 24/24 passed provenance validation, and the final unique candidate set is 58/58 valid. Original training count remained zero.

No epoch smoke was used, and completed canonical task keys were reused rather than retrained. Under the original preregistered rule, every candidate failed its discrimination gate and none qualified for expansion. The later RTDR-only audit was a separately reported, user-directed post-gate stability study; it did not change the initial FAIL and did not satisfy full-audit `stable_positive`.

## Automatic publication

Training, artifact collection, and provenance validation are complete for 40 Original controls and 58 unique candidate archives. Evidence commit `97370fd49cb130bc10c620f1293ebff00985b729` was pushed to [YB123-DT/git_gcnet](https://github.com/YB123-DT/git_gcnet) as `exp/second-graph-aggregators`, and `git ls-remote` returned the same SHA. Before publication, 260 tests passed with one expected skip and all 246 checksum-ledger entries verified. Dataset/features, credentials, caches, and diagnostic-only failed attempts were not published as canonical results. This later documentation-only attestation does not alter the archived evidence.

Trilingual evidence: initial [RESULTS](results/RESULTS.en.md) and [ANALYSIS](results/ANALYSIS.en.md), bounded [RTDR extension](results/rtdr_extension/RESULTS.en.md), and [RTDR full audit](results/rtdr_full/RESULTS.en.md), each with canonical and Chinese mirrors plus machine-readable summaries.
