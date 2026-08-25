# GCNet Second-Graph Experiment — English Record

Canonical detailed protocol: [EXPERIMENT.md](EXPERIMENT.md). Chinese record: [EXPERIMENT.zh.md](EXPERIMENT.zh.md).

## Scope and current status

This experiment changes GCNet's graph mechanism while inheriting the 40 existing IEMOCAPSix fold-5 Original archives. Original is not retrained.

| Candidate | Insertion | Parameters | Status on 2026-08-25 |
|---|---|---:|---|
| GenAgg | Both second-layer `GraphConv` aggregators | +118 total | Implemented; its six-job share of the combined 12-job Phase A is pending |
| Scaled Soft Medoid | Both second-layer `GraphConv` aggregators | +0 | Implemented; its six-job share of the combined 12-job Phase A is pending |
| SSMA | Both second-layer `GraphConv` aggregators | +595,400 total | Implemented; official-environment gate passed; formal Phase B pending |
| Custom RTDR | Planned two-hop relation-transition routing | Expected +0, unverified | Implementation, equivalence tests, and training pending |

The reason for GenAgg is to learn cardinality, centering, and nonlinear set behavior that fixed sum cannot represent. Soft Medoid tests robustness to an isolated message-space outlier. SSMA explicitly models cross-neighbor products before compression. RTDR will test whether deleting off-diagonal two-hop relation transitions is useful; it must not be described as MrMP/MMP transfer because the inspected [MrMP paper](https://arxiv.org/abs/2202.04844) mixes relations within each layer.

Primary sources are [GenAgg](https://arxiv.org/abs/2306.13826), [Soft Medoid](https://arxiv.org/abs/2010.15651), and [SSMA](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html). The complete source/version trace and exact formulas are in [the canonical record](EXPERIMENT.md).

## SSMA official-environment evidence

Commit `08aa55fb255d5e32aa9f6171246e6e2821c97c71` was checked on biggpu with Python 3.8.20, Torch 1.8.0, and PyG 2.0.1. CPU and GPU FP32 forward/backward outputs and gradients were finite. The measured extra parameter count was 297,700 per branch. A synthetic 64-node GPU check peaked at 56,989,696 allocated bytes and 92,274,688 reserved bytes. Torch emitted a complex-to-real warning, but gradients remained finite. This is compatibility evidence only, not training or accuracy evidence.

## Two-phase locked grid

- Phase A: `{genagg, soft_medoid}` × missing `{0.0, 0.7}` × seeds `{66,67,68}` = 12 jobs.
- Phase B: `{ssma, rtdr}` × the same rates and seeds = 12 jobs, after RTDR passes its required equivalence gates.
- Both phases use `stage=formal`, fold 5, GPUs 0–3, and three workers per GPU. The total is 24 new candidate jobs and zero Original jobs.

The 40 read-only Original NPZ files under `/data2/yb/paper/experiments/cp_lecc_iemocap6_20260824/protocol_recovery_v1_biggpu/formal/original` will be joined by rate, seed, fold, and mask SHA256. Before candidate launch or comparison, a dedicated legacy-aware Original validator must still be implemented, tested, and pass. It may map only historically absent fields to the locked defaults `addition/bilstm/bilstm/add`; source, command, dataset, features, parameter count, rate, seed, fold, and mask provenance must remain strict. This is a pending gate, not a claim that the validator already exists.

No epoch smoke is allowed. Each candidate receives one official-environment synthetic forward/backward check; the shared runner then launches a bounded 12-job wave. Completed task keys are resumed, not rerun. A scientific negative result is still preserved, while a corrupted or incomplete artifact remains `INCOMPLETE`.

## Automatic publication

After both waves and provenance validation finish, the workflow automatically synchronizes code, task-level results, manifests, hash records, and trilingual result summaries back to `/data2/yb/paper`; runs tests on the publish tree; makes a Lore-format commit; and pushes the verified commit to [YB123-DT/git_gcnet](https://github.com/YB123-DT/git_gcnet) as `exp/second-graph-aggregators`. It then compares the remote ref with the local commit. Dataset/features, credentials, caches, and incomplete artifacts are not uploaded. A push failure retries publication from the same commit and never reruns training.

Formal metrics and RTDR implementation remain **PENDING**; no result is inferred here.
