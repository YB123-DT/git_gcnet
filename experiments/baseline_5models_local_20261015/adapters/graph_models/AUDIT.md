# SDR版本适用范围

本文下述SDR实现修复仅适用于`implementation_fixed`，该版本独立保留用于比较，不能代填主文成绩。主文使用`released_compatible`：只允许MELD输入1666维/238个可整除Attention头和九说话人81种关系这两类数值兼容，以及包隔离的相对导入。Released保留官方Tuple拼接后仅Temporal分类、原MHA轴、原重建Padding展平行为，不将这些上游问题静默修复。来源、完整Diff及等价测试见`sdrgnn_released/`；IEMOCAP保持2560/256、原两说话人关系路径并在Python启动前固定hash seed。

# GCNet / SDR-GNN isolated implementation audit

No parent repository files were modified; no training, GPU use, upload or remote write was performed in this audit. Python files are snapshots copied from the source repositories below, followed by the recorded patches. Source status and per-file hashes are in each `provenance.json`.

## Source identity

- GCNet: local `/data2/yb/paper/GCNet_TPAMI`, origin `https://github.com/CASIA-Affective-Computing-Group/GCNet.git`, SHA `6c7909056926b9e5b746366b80f3b27e75fe7db0`. Additional `github` remote points to `YB123-DT/git_gcnet`. Root config.py has preexisting modifications; copied graph model files were clean. The origin remote differs from the requested zeroQiaoba URL, but the requested official reference has now been independently checked below; all four consumed original source files are byte-identical.
- SDR-GNN: local `/data2/yb/paper/SDR-GNN_official_2024`, origin `https://github.com/fufangze/SDR-GNN.git`, SHA `b45546111e9dbe50e019cfa98a6c19f13a4c8aa8`, clean tracked working tree.

## GCNet compatibility changes

1. Replace the hard limit of two speakers with deterministic `N*N` directed speaker relations. IDs use ordered `i:j` pairs; nine MELD speakers yield 81 relations, without compressing identities. Temporal relation order is fixed to past/now/future. Edge ordering is sorted. Previously Python set iteration made relation numbering depend on hash seed. Old checkpoints require their original relation mapping; these changes do not assert compatibility with unidentified legacy mappings.
2. Package-relative imports avoid collisions between the two upstream modules both named graph.py.
3. Correct the unused upstream GRU branch's undefined `U` reference to `inputfeats`. The selected official LSTM branch is unchanged.
4. The loss helper aligns validity with time-major reconstruction. Upstream flattened T,B features against an untransposed B,T padding mask; the same modality-normalized missing-feature MSE is preserved with correct validity alignment.

The base bidirectional LSTM, relation GCN, GraphConv, bidirectional graph fusion LSTM, optional matching attention, temporal future window, reconstruction and classifier remain. MELD name: **GCNet MELD-adapted Original**. No causal claim.

## SDR-GNN official defects and explicit fixed-version name

1. `GraphNetwork.forward` returns `(hidden,)`. Adding the two graph outputs concatenates tuples; `hidden0[0]` discards the speaker graph. Reproduced before repair: temporal conv receives classifier gradient; speaker conv gradient is None. Change the return to Tensor and sum both Tensor outputs before classifier/reconstruction. Both graphs now receive nonzero classifier gradients. This changes the actual upstream behavior and must be named **SDR-GNN Implementation-fixed**, with MELD-adapted suffix for MELD. Do not label it as an unmodified Original.
2. MultiheadAttention uses default sequence-first mode, but upstream transposes T,B,D to B,T,D and mixes dialogues as the sequence axis. Keep T,B,D and supply utterance padding key mask. No graph, hypergraph or attention module is deleted.
3. Upstream hardcodes attention embedding width 2560 and 256 heads (the IEMOCAP feature total). Set embedding width to A+T+V and retain the largest divisor no greater than 256 heads. Thus IEMOCAP retains 256 heads/2560 features; MELD 1666 features uses 238 heads (7 features/head). This is necessary dimensional compatibility, explicitly recorded rather than silently projecting input features to a different feature space.
4. Apply the same deterministic multi-speaker graph mapping and loss validity alignment as GCNet.

**Retained forward path:** bidirectional GRU/LSTM -> relation GCN -> HypergraphConv -> highConv -> bidirectional graph GRU -> temporal/speaker sum -> classifier and reconstruction. Reconstruction is followed by self-attention and a linear hidden representation, just as upstream. Classification uses the graph-fused hidden *before* reconstruction attention; `reccls_flag` remains false per official default. The wrapper accepts only masked observed inputs. Full features enter training loss targets only.

**Hypergraph audit:** official custom HypergraphConv runs successfully under PyG 2.4.0, including both source-to-target and target-to-source propagations; no replacement convolution is introduced. The official code passes ordinary graph edges as hyperedge incidence indices. It also defines some parameters (`weight`, `edgeweight`, `edgefc`) unused by the default non-attention branch. These upstream design choices remain and must not be described as newly learned projections. No unverified algorithm rewrite was made.

**Inherited limits:** upstream bidirectional recurrent encoders process padded sequences without packing, so changing padded sequence length may affect valid outputs; this is retained architecture, not claimed padding-invariant. Batch-independence tests change another dialogue while holding the padded sequence length fixed. Neither graph model is causal.

## Official IEMOCAP training configuration

From GCNet README: epochs=100, lr=.001, hidden=200, windows=2/2, LSTM, reconstruction enabled, batch=32, dropout=.5, Adam weight_decay=1e-5. Loss CE + reconstruction.

From SDR-GNN README/train.py: epochs=100, lr=.001, hidden=200, windows=3/3, GRU, reconstruction enabled, recon_loss_weight=.5, batch=32, dropout=.5, Adam weight_decay=1e-5. Loss .5*CE + .5*reconstruction. Four/six classification outputs and two speaker IDs; same A/T/V feature order. Five-session folds are supplied by common_data/runner rather than the upstream random-fold dataloader. No claim that upstream's fold allocation already meets this task.

## Runner interface and environment

`from adapters.graph_models import build_model`

`build_model(config, dims, nclasses, nspeakers)` where config has `model` GCNet/SDR-GNN; dims is (A,T,V) or dictionary a/t/v. Optional keys: hidden, graph_hidden, base_model, window_past/window_future (windowp/windowf aliases), dropout, time_attn, loss_recon, recon_loss_weight.

`model(masked_x_TBD, speakers_BT, valid_BT, lengths)` returns dict logits TBC, reconstruction TBD, hidden TBH. `model.loss(outputs, labels_BT, full_training_target_TBD, availability_TB3, valid_BT)` returns `(total_loss, {'classification':..., 'reconstruction':...})`.

Dispatcher must bind one GPU UUID per process; original .cuda() internals then map to cuda:0. No global CUDA setup or logger upload is performed by wrapper. Use `/home/yangbin/miniconda3/envs/multimodalerc310/bin/python`: existing torch 2.2.2+cu121, torch_geometric 2.4.0 and torch_scatter 2.1.2+pt22cu121 imported successfully. comp-repro lacks PyG/scatter. No new dependency was installed.

## Verification

CPU red/green regression: unpatched copies failed nine-speaker forward for both models and speaker-branch classification gradient for SDR; repaired copies pass the same regression. Three test methods exercise 9 speakers/81 relations, 7-class forward/backward for both models, both SDR graph gradients, sequence-first attention, finite reconstruction, masked objective, and graph/attention batch isolation. This is implementation validation only; no score or completed training result is claimed. Check `verification.log` for actual output.

## Requested zeroQiaoba source equivalence verification

Read-only verification on 2026-10-15 resolved `https://github.com/zeroQiaoba/GCNet.git` HEAD to **8cce4a0c9a50172abfa79971e77c71c40c1d733d** using `git ls-remote`. The four `gcnet/` files were fetched through raw.githubusercontent.com with that exact SHA pinned in each URL and compared with the original local source files and vendored upstream provenance hashes. All four match byte-for-byte, including line endings; there is no substantive difference in the audited model, relation graph, attention helpers or loss code. No extra repository clone is needed. This establishes these components' equivalence, not an assertion that every repository file or entire Git history is identical.

| File | Reference and local original SHA256 | Result |
|---|---|---|
| `gcnet/model.py` | `3e8b890be4485405157eb92fbf106896555b92b8aee1de0e07155454b89aa038` | Byte-identical |
| `gcnet/graph.py` | `24edbf2b4734c84a41c9bb32165802a88a9a840965540c507229f195332a72a1` | Byte-identical |
| `gcnet/module.py` | `5bf0494153a364981684b201cd8585590ad0f894fa5d8a55054cdb93bf16de7b` | Byte-identical |
| `gcnet/loss.py` | `7c4f69d716eb769c4ca7e24aaff95dbc0a5f746986701240dc0395c59363c9bb` | Byte-identical |

The pre-existing local origin/SHA and local adapter patch remain recorded separately. `gcnet/provenance.json` includes pinned reference URLs, verification time and matching checksums. No implementation was changed by this verification.
