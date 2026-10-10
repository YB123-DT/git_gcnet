# Training-time gradient monitoring

INTERNAL DIAGNOSTIC ONLY. First bounded paired run: MOSI seed66 original Flat
versus original Nested,100epochs each, from scratch. No multi-seed expansion.
Use original common source_ad211c0, exact Flat configuration, change only Nested
switch. Same task MSE, Adam LR.001, batch32, cyclic missing masks, clipping1,
100epochs and per-rate Test-oracle selection. Healthy biggpu GPU7, two parallel
jobs, independent /data1 outputs; existing results are not overwritten.

Observer hooks are attached after original initialization; no RNG draws,
extra forward/backward, tensor replacement, new optimizer group or loss.
Record training-only input gradients at readout boundary: Local includes original
Skip and adapter paths; Base/Gap use actual512forward dimensions. Exclude first
turns for Memory and inactive Gap/padding. Local keeps all valid turns.
Record native sample-mean loss gradients and gradients scaled by valid sample
count, to expose the native batch denominator. This is not output Jacobian.
Do not compare native batch-mean numbers directly with prior detached per-sample
sum-gradient diagnostics. Full causal BPTT remains intact during training.

Before invoking original clipping, measure global parameter gradient norm and
group L2/RMS for Nested, Flat adapter, Local Skip/path and observed encoder.
Record expected clip coefficient. Group L2 depends on parameter count; RMS is
provided as another description, not proof of optimization quality.
Nested residual/unchanged input norm per Local/Base/active Gap is recorded at
actual task-side block output, not interpreted as memory reliability.

Save small per-step aggregate records in gradients/epoch_NNN.json, tagged with
epoch/rate and exact availability hashes. No full gradient tensors are saved.
Only two training batches per epoch in this MOSI split: not all rates appear
every epoch. Compare identical available epoch/rate pairs, never fill missing
rates with zeros; aggregate across a full8epoch cyclic window if needed.

Tests cover exact output/parameter-gradient/RNG parity on a small real autograd
graph, Local Skip, inactive Gap/padding/first-turn filtering, residual ratios,
finite group norms, and no evaluation logging. Instrumentation is isolated from
the historical model source and other running tasks. Preserve all BEST and full
recovery artifacts. Finish checks require100epochs,100observer files and masks.
