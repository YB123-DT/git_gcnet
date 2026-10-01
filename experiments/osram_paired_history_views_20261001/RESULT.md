# Paired-history results — seed66

INTERNAL TEST-ORACLE DIAGNOSTIC, not formal validation-selected paper results.
Code32c6897; launch526d8bf. Both100epochs completed on biggpuGPU6.
Original Flat reused; eight per-rate best checkpoints per arm.

W-F1(%), deltas in percentage points. Single seed only; no significance claim.

|Missing rate|Flat|Paired task-only A|Paired InfoNCE B|A−Flat|B−Flat|B−A|
|---|---:|---:|---:|---:|---:|---:|
|0.0|88.205|87.102|86.863|-1.103|-1.342|-0.239|
|0.1|86.507|85.715|85.566|-0.792|-0.940|-0.148|
|0.2|83.187|82.041|82.565|-1.146|-0.622|+0.524|
|0.3|80.763|80.520|80.163|-0.243|-0.600|-0.357|
|0.4|80.827|80.130|79.625|-0.697|-1.201|-0.504|
|0.5|77.494|77.112|76.271|-0.382|-1.223|-0.841|
|0.6|75.790|75.483|75.328|-0.306|-0.462|-0.155|
|0.7|75.773|74.987|75.756|-0.786|-0.017|+0.770|
|8-rate mean|81.068|80.386|80.267|-0.682|-0.801|-0.119|
|High(.5/.6/.7)|76.352|75.861|75.785|-0.491|-0.567|-0.076|

## Training exposures

|Arm|Anchors|Usable anchors|Removed observed bits|View1 observed bits|Actual drop|Batches without negatives|
|---|---:|---:|---:|---:|---:|---:|
|contrastive|72304|72304|42935|265448|16.1745%|0|
|control|72304|72304|42935|265448|16.1745%|0|

Counts are summed training exposures over100epochs, not unique utterances.
Both arms match View1 AND View2 mask hashes at every epoch. Evaluation masks
verified against original Flat by the run verifier. Task loss is original MOSI MSE;
W-F1 is the primary outcome. Fixed drop.2/lambda.1/temp.1; no hyperparameter search.
No original model/backbone changes; projector is training-only. Same-conversation
nonpositive anchors excluded. More compute than single-view Flat; A is the key
control for attributing changes to contrastive training rather than two-view augmentation.
No fixed-pattern inference or downstream module search was added.

## Interpretation

Task-only versus Flat:8-rate-0.682pp; high-0.491pp.
InfoNCE versus task-only:8-rate-0.119pp; high-0.076pp.
This fixed seed/configuration provides no overall improvement. It does not establish
that all contrastive/history-invariance methods fail, nor identify the causal reason.
No additional runs or hyperparameter changes were made after observing these scores.
