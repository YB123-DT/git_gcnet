# MMIN source and adaptation audit

## Provenance

- Official: https://github.com/AIM3-RUC/MMIN, `master`, commit `69a45fedff66fa0da74205ad425440c1ee8f6d1f`. Default `main` contains no implementation; the official README directs users to `master`.
- Read-only shallow clone: `adapters/mmin/upstream/`; clean after download, no upstream changes or uploads.
- Existing implementation: `/data2/yb/paper/GCNet_TPAMI/baseline-mmin/`, repository https://github.com/YB123-DT/git_gcnet, commit `6c7909056926b9e5b746366b80f3b27e75fe7db0`; MMIN source has no uncommitted modification at audit time. Current branch and file SHA256 are in `adapters/mmin/provenance.json`.
- Adaptation diff: `adapters/mmin/official_to_gcnet_utt.diff`, comparing existing UTT implementation with official `master`.

The official method consumes frame features with audio/visual LSTM encoders and a text CNN. The existing GCNet baseline replaces those encoders with `FcClassifier` on pre-extracted utterance vectors. The wrapper preserves that existing feature adaptation, rather than pretending that frame-level encoders were used. Report the implementation as **pre-extracted-UTT adapted Original**; for MELD, **MELD-adapted Original**. No history encoder or dialogue network is added. The teacher pretraining, copied modality encoder initialization, residual AE, complementary-modality reconstruction objective, cycle objective, and emotion classifier remain present.

Official and existing GCNet `FcClassifier` and `ResidualAE` classes have identical parsed ASTs. `adapters/mmin/networks.py` vendors these exact class definitions with the MIT license so run snapshots do not depend on mutable external source directories. `netT` is the wrapper's explicit text name for upstream `netL`. Public masks are A/T/V; upstream uses A/V/L, so direct raw mask interchange is unsafe.

## Training and evaluation boundaries

`Teacher(config)` uses full **training** utterance features. Its artifact is selected at the final pretraining epoch, independently of test performance. Its immutable reuse identity requires dataset, fold, training seed, feature content hash, training split hash, teacher budget and complete config, and deliberately excludes missing rate. `save_teacher` refuses incomplete artifacts and existing destination files. `load_teacher` verifies identity/configuration and loads strictly. These teacher artifacts are weights for reuse; full optimizer/RNG/progress state for interrupted-run resumption belongs in the main runner checkpoint.

`MMIN(config, teacher)` copies the teacher encoders and freezes a separate teacher snapshot. `forward(a,t,v,visible)` only accesses masked input features and never invokes the teacher. Absent features are removed with `torch.where`, which also prevents NaNs in unavailable entries from propagating. The complementary missing training features are read only in `losses` while the model is in training mode, matching upstream reconstruction supervision. Teacher supervision never becomes a classifier input. During evaluation neither labels nor teacher features are required.

The formal missing-model checkpoint is chosen by **Test ACC**, earliest epoch wins ties, under the shared fixed test mask. ACC/WF1/UA/Macro F1 must all come from that checkpoint's same predictions. This is `checkpoint_selection=test_best_acc`, `evaluation_protocol=test_oracle`: test-based model selection causes optimistic selection bias and is not an unbiased independent evaluation. Selection, saved predictions, fixed masks, and fold/seed aggregation are owned by the common runner, rather than independently reimplemented in this adapter.

## Frozen starting hyperparameters

Exact machine-readable settings are in `adapters/mmin/hyperparameters.json`.

| Setting | MELD | IEMOCAP-4 / IEMOCAP-6 |
|---|---|---|
| Teacher epochs | 100 | 20 (official 10 constant + 10 decay) |
| Missing epochs | 100 | 60 (official 30 constant + 30 decay) |
| Batch / LR | 256 / 1e-3 | 128 / 2e-4 |
| Embeddings / classifier | 128 per modality / 128,128 | same |
| Residual AE / blocks | 256,128,64 / 5 | same |
| Teacher / missing dropout | .3 / .5 | same |
| CE / reconstruction / cycle | 1 / 4 / 2 | same |
| Teacher Adam / missing Adam | beta1 .9 / .5; beta2 .999 | same |
| Batch norm | false | false |
| Initialization | normal, std .02 | Kaiming normal, fan-in, a=0 |
| Teacher / missing gradient clip | 5 / 5 (existing GCNet adaptation) | .5 / 1 (official) |

IEMOCAP settings are grounded in official `scripts/CAP_utt_fusion.sh` and `scripts/CAP_mmin.sh`. That official experiment uses 10 folds and different frame features; this task instead mandates the shared five session folds and shared UTT features. Six-class output is a compatibility extension of the official four-class setting. MELD uses the user's requested 100/100, batch 256, LR1e-3 with the existing GCNet script's 20 constant / 80 decay schedule. Teacher MELD uses the same 20/80 schedule as an explicit adaptation decision. The teacher is reused across missing rates only when its identity matches.

The official missing script declares `weight_decay=1e-5`, but `MMINModel` does not pass it to Adam, so the effective upstream weight decay is **0**. This is preserved. Missing Adam defaults to beta1=.5 in upstream options; teacher explicitly sets .9. Initialization and clipping differ between the official and existing GCNet configurations. MELD follows the existing GCNet configuration: normal linear weights (std .02), per-network teacher/missing clipping 5/5. IEMOCAP follows official options and model code: Kaiming normal (a=0, fan_in), per-network teacher/missing clipping .5/1. Official `init_gain=.012` is recorded but does not affect Kaiming initialization. Linear biases are zero in both configurations. These choices are explicit `MMINConfig` fields and form part of the teacher artifact identity.

## Verification

CPU command: `/home/yangbin/miniconda3/envs/comp-repro/bin/python -m unittest discover -s /data2/yb/paper/experiments/baseline_5models_local_20261015/tests -p test_mmin.py`.

Eight tests passed: teacher optimization and copied initialization; hidden feature perturbation including NaN plus corrupted teacher invariance during prediction; gradients and parameter updates for all six missing-model networks while teacher stays frozen; exact teacher save/load and cross-MR construction with mismatched identity/overwrite rejection; one-utterance output and eval batch independence; invalid-mask rejection; official IEMOCAP Kaiming initialization and per-network teacher/missing clipping; unsupported-initialization rejection.

No GPU training, data result reuse claims, performance scores, or full-run completion are asserted by this adapter audit. Source and CPU contract verification are complete; the common runner must still execute real data smoke tests and formal training and save their actual artifacts.

The same eight tests were subsequently run in `/home/yangbin/miniconda3/envs/multimodalerc310/bin/python` (PyTorch 2.2) and passed in 0.975s. That environment exposed PyTorchFileWriter rejecting hidden temporary basenames; immutable teacher publication now writes a `teacher-*.pt` temporary file before `os.link`, retaining the no-overwrite guarantee. The save/load/overwrite test exercises this real serialization path.

## Exact learning-rate epoch mapping

Both official scripts default to `epoch_count=1`. The loop trains epochs 1 through `niter+niter_decay` inclusive. `LambdaLR` is constructed before training; with PyTorch 1.12 it immediately initializes `last_epoch=0`. The runner calls `scheduler.step()` after training/checkpoint saving and before evaluation at the end of each epoch. Upstream lambda is `1 - max(0, scheduler_epoch + epoch_count - niter)/(niter_decay+1)`. Thus the learning-rate factor **used to train epoch k** is `1 - max(0,k-niter)/(niter_decay+1)`. Teacher 10/10 uses base LR in epochs 1–10, 10/11 base in epoch 11, 1/11 base in epoch 20; missing 30/30 uses base in epochs 1–30, 30/31 base in epoch 31, 1/31 base in epoch 60; MELD 20/80 uses base in epochs 1–20, 80/81 base in epoch 21, 1/81 base in epoch 100. The final end-of-epoch step reaches zero after training is already complete. These mappings were checked by executing the actual LambdaLR on the installed PyTorch.

Cycle target is exactly the upstream `feat_fusion_miss.detach()`; reconstruction target is the frozen teacher encoding of complementary missing modalities; no teacher emotion-label probability target or extra teacher label coefficient exists. MMIN consumes flattened real utterances; the shared runner must omit padded utterances from both training loss and predictions. No extra modality-classification mask is needed, and learned encoder biases on all-zero missing inputs match upstream behavior.
