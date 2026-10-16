# Frozen evaluation protocol

- Server: local user23 only; no network logging or uploads. CPU threads=4 per job. MAX_GPU_JOBS=2, one process per GPU.
- Whitelist: GPU-7ff18ebb-100e-98fe-038b-a98bd4f94248 and GPU-eed2f34f-b4c0-7b7a-498e-71a026e30c5a. Recheck compute PIDs/utilization before launch. Never touch unrelated processes.
- Seeds=66,67,68; nominal MR=0,.1,.2,.3,.4,.5,.6,.7, retain >=1 modality. Record measured missing fraction (MR .7 saturates at 2/3).
- Fixed test mask keys: dataset, dialogue ID, utterance ID, canonical MR, evaluation seed 20261015. Same across baselines/training seeds/epochs/batching.
- MELD original ECERC train/valid/test splits, 7 labels, 9 speaker IDs, A/T/V identical for every model. Valid diagnostics optional.
- IEMOCAP: 5 independent leave-session-out folds, remaining four sessions train. Pool true per-fold selected-epoch predictions within seed, then sample std/variance (ddof=1) across exactly 66/67/68. Never count folds as seeds.
- checkpoint_selection=test_best_acc; evaluation_protocol=test_oracle. Strict improvement selects earliest tied epoch. All metrics from the same prediction artifact. Two-stage selection restricted to Stage2 (epoch>stage_epoch).
- Test-based checkpoint selection introduces optimistic test selection bias; these are not independent unbiased or leakage-free test estimates. No hyperparameter search by test score.
- Historical dynamic-mask results remain LEGACY. A single historical best checkpoint cannot reconstruct fixed-mask epoch-oracle selection; distinguish checkpoint reevaluation from a complete oracle rerun.
- Training masks sampled online with common distribution; no hidden test modality or labels can feed model prediction. Reconstruction targets may use only training data.
- MoMKE causal/original differ only in temporal mask; official experts/routing/fusion retained. 100/50 vs 200/100 diagnostics are assessed using training loss stability/convergence only. Config remains gated until decision recorded.
- ComP uses existing causal model without redesign. Original uses its noncausal flag and prompt path. MMIN UTT encoders preserve existing pre-extracted-feature adaptation, teacher cached per dataset/fold/seed/config. GCNet remains bidirectional. SDR released_compatible is the main baseline, retaining tuple graph selection, attention axes and released reconstruction padding. Only input-width/head divisibility and multi-speaker relations are adapted for MELD; implementation_fixed remains a separate comparison and cannot fill missing released results.
