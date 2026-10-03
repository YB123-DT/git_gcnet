# Current–History Relation residual: seed66 screening

INTERNAL DIAGNOSTIC ONLY

Approved scope: one MOSI seed66, 100 epochs, original cfg84 cyclic 0.0–0.7
single-view random missing, original Adam/lr/batch/loss, per-rate BEST test-oracle.
No validation-selected paper claim. No automatic multi-seed expansion.

Baseline implementation before this change: `7d56881f13e50b98a65a8fc9ca19f31222b39a55`.
Reference run on **biggpu**:
`/data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66`.
Its saved config, metrics, masks and predictions are the protocol reference;
the reference model is not retrained. Current worktree branch is
`feature/osram-uniform-forced-text` (not the older handoff branch name).

## Fixed design

Keep original Flat and Local Skip and task head; add residual before its existing
emotion LayerNorm. Only the forward half of each read enters the new block:
q=Linear(Local), k_i=shared Linear(C_i), both dimension128.
z_i=[q,k_i,q*k_i,abs(q-k_i),type_i16]. Shared LayerNorm → Linear128 → GELU →
Dropout(original cfg84 p=0.5) → Linear64. Active-count mean → zero-initialized
Linear1600. No Memory formula or query changes, no auxiliary loss.

Base valid, Gap only for its missing modality; inactive inputs and outputs safezero.
First valid utterance strictly skips the residual using prior valid-position count,
including after training. Padding is zero. New initialization is RNG-isolated.
New feature defaults off and original checkpoints remain usable with it off.

Capacity control is implemented/tested, not trained: q and active pooled k →
ordinary MLP →64→same zero output. Hidden width chosen for near parameter matching;
no product/difference pairwise features. Parameter counts saved by run.py.

## Execution / verification

Remote isolated snapshot:
`/data2/yb/remote_experiments/osram_current_history_relation_20261003/code`.
Use host GPU5, UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`; GPU4 forbidden.
Run after correctness checks with:

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
GCNET_DATASET_ROOT=/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset \
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
experiments/osram_current_history_relation_20261003/run.py \
--output /data2/yb/remote_experiments/osram_current_history_relation_20261003/seed_66 \
--gpu 5 --commit CODE_COMMIT
```

Checks: off-path/state/RNG, initial full-model equality, first utterance even with
trained output bias, inactive NaN poisoning, padzero, finite gradients/updates,
forward-half-only, masked ablation inputs and unchanged scan, CLI/load plumbing.
Final verification also checks all eight canonical masks and BEST checkpoints.

## Fixed Context Audit

Reuse `osram_context_audit_20261003/adjacent_results/utterances.csv` seed66 rows.
Near=abs(original Flat Local-only prediction)<=0.25; far otherwise.
Same/opposite=immediate adjacent nonzero gold polarities, never model inputs.
No cross-conversation pairs or skipping neutral utterances.
Compare old Flat full vs relation full on identical cells and availability.
Compute W-F1 per rate then macro; sum correction/harm counts as sample appearances,
not independent sample counts. No new group assignment from the new model.

Only full-test improvement with retention of near+same benefit warrants further
work; a selected cell gain alone is not success. Single-seed findings have no
multi-seed uncertainty estimate. Norm diagnostics are internal statistics, not
conflict, reliability or emotion-shift estimates.
