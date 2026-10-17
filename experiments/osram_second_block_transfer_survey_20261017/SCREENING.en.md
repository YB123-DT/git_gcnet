# Second-block screening: alternative mechanisms

Method sections and key implementations inspected; not an exhaustive full-paper
review or migration validation. No training or model edits. No direction selected.

| Candidate | Retained mechanism | Main boundary |
|---|---|---|
| MultiModN | Available modalities update a shared task state; missing modules are skipped. | Unlike prior Local/Base/Gap logit corrections,but related to multi-exit supervision. MNAR benefits do not establish gains under GCNet random masking. |
| TabM | Parameter-sharing ensemble with member losses and prediction aggregation. | Not MoE or multiple Memory scans. Changes prediction/loss interface; reduced parameter count does not eliminate compute overhead. |
| Hyper-Connections | Multiple residual streams with learned across-depth/width connections. | Related to cross-stitch/Mixer. A one-layer mixer is not a faithful port; deep-network motivation is unestablished for shallow Flat. Lower priority. |

Sources: [MultiModN](https://arxiv.org/html/2309.14118),[TabM](https://proceedings.iclr.cc/paper_files/paper/2025/file/c1ba41c694834aeef91ae161711d4939-Paper-Conference.pdf),[Hyper-Connections](https://proceedings.iclr.cc/paper_files/paper/2025/file/f1e8ff97057e0c3f6bcb7018c78d4df1-Paper-Conference.pdf).

Code caveats: MultiModN's batch-wide any-NaN skip is unsuitable for mixed
per-utterance masks; its objective also includes a state-change penalty. TabM's
shared weight/member scaling and member-versus-aggregate objective were checked.
For Hyper-Connections,only author-provided appendix pseudocode was verified.

The registry,meaningful/core20 implementations,decision correction and
modality-track/CED results were searched. Absence of exact names is not proof
of novelty or exhaustive semantic deduplication. User selection is pending.
