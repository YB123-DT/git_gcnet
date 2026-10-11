# Evidence review and candidate directions (not approved experiments)

INTERNAL DIAGNOSTIC ONLY. User requests a better direction after repeated Nested failures. No new training or model changes authorized by this review; the existing approved150-epoch pair remains running.

## Evidence checked

- Small Flat three-seed:79.163→79.301 with Nested, only+.138pp; seed67 declines1.001pp. Source ../osram_small_flat_3seed_20261009/RESULT.md. Earlier seed66-only positive statements must not be generalized to robust gains.
- Frozen rooted probe: Flat79.654480, Flat+Rooted79.058842, Flat+PreDecoder78.396285. Source ../osram_nested_root_probe_20261011/completed/RESULT.md. Three probe seeds on one backbone, not three backbone seeds. No evidence of incremental task benefit under this probe; not an information-theoretic impossibility.
- Frozen Flat+Nested:79.540→79.353 under its matched one-checkpoint protocol; do not compare these directly with per-rate Flat81.068. Source ../osram_frozen_nested_20261009/RESULT.md.
- Warmup/cosine seed66: Flat81.068→80.029, Nested80.992→80.323. Source ../osram_nested_cosine_20261010/RESULT.md. A different schedule is not automatically beneficial.
- Lower-LR150 three-seed MOSI Flat80.747 vs Nested80.562; Flat67's gain comes from five rates, primarily.6/.7. Source ../osram_complete150_20261011/RESULT.md.
- Memory probe real+Local79.785 exceeds Local78.082 but not original Flat81.068; subsequent residual correction failed to exceed Original/Local/Donor. Sources ../osram_frozen_memory_audit_20261009/RESULT.md and ../osram_frozen_residual_20261009/RESULT.md. “Memory contains usable signal” is not evidence that a more complicated readout will improve original Flat.
- Historical write protection has also been attempted; association preservation alone did not establish superior task benefit over norm-matched global intervention. Do not recommend it as an untested discovery. Source ../osram_write_intervention_20260909/RESULT.md.

Live snapshot during this review: concat76/100 initial stage, cumulative per-rate BEST mean66.315196/high65.185976; matched control100/100 initial stage,80.609498/high76.335941. Both have planned50 additional lower-LR epochs. These are interim scores, not final150 results. No NaN/mask/gradient cause is established; poor task training is not proof of an implementation bug or universal failure of concat.

## Two candidate directions

1. Conservative: retain existing raw features and Flat; bounded optimization robustness work, based on the actual lower-LR continuation results. No claimed novel Block, no claim cosine universally helps. Lower risk/resource demand; limited methodological novelty and no guarantee beyond seed67.
2. Recommended research pivot, subject to user choice: task adaptation of ORIGINAL observed audio/visual feature extractors, retaining OSRAM and Flat downstream. This is not another projector operating on the same fixed utterance vectors, missing-latent completion, or text prediction. Start one modality/one fixed adaptation setting and compare frozen vs train-adapted encoder under identical downstream protocol. Requires raw recordings, split-safe train-only adaptation and resource review. Broader scope; current failures do not prove input information insufficiency. A stronger encoder alone is not a new research contribution; any missing-modality-specific method needs separate justification and controls.

Primary-source feasibility pointers (not proof of MOSI gains and not a committed architecture choice):
- https://arxiv.org/abs/2110.06309 — Exploring Wav2vec2.0 fine-tuning for improved speech emotion recognition.
- https://arxiv.org/abs/2405.04485 — Adapting WavLM for Speech Emotion Recognition.
- https://openaccess.thecvf.com/content/CVPR2023/html/Cai_MARLIN_Masked_Autoencoder_for_Facial_Video_Representation_LearnINg_CVPR_2023_paper.html — facial video representations and downstream tasks.

Recommendation is about changing the information/representation available to the backbone, not promising a score or claiming novelty. No raw-data acquisition, encoder fine-tuning, new loss, extra GPU job or final publishable angle has been selected. Await whether user accepts moving upstream versus retaining frozen-feature constraints.
