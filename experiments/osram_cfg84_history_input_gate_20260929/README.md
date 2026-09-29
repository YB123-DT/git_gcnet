# cfg84 Flat + bounded scalar history-input gate

User-authorized implementation and three-seed training, 2026-09-29.
Internal Test-oracle comparison, not a validation-selected paper result.

## Fixed architecture

C = concat(emotion_base, availability-masked emotion_gap[A,T,V]).
G = Linear(latent_dim + 4*context_dim + 3,128) -> ELU -> Linear(128,1).
alpha = 1 + .2*tanh(G(concat(Local,C,availability))).
Feed concat(Local, alpha*C) to the original Flat emotion_adapter, retaining
the original local_skip, emotion_norm and task head. One scalar per utterance.
Last G linear weight/bias are zero initialized, so alpha starts at one.
No extra normalization, Dropout, GRN, vector gate, predictor or auxiliary loss.
The new flag --osram-history-input-gate defaults off and keeps flat readout.
Local, Memory writes/reads, queries and keys/values are not modified.
Input masks use fixed Gap slots; conditions cannot bypass emotion ablations.

## Training contract

Use original cfg84 no-JEPA reference configs for seeds66/67/68, 100 epochs,
cyclic random missing, emotion-only, forward-only OSRAM, output_dim1600,
8 heads/key64/value64/write_step.6. The sole experimental config delta is
osram_history_input_gate=True. From-scratch joint training: neither original
Flat nor Memory is frozen. No old task checkpoint initialization and no
test-label oracle/correction table supervision. Baseline results reused.

Keep existing per-rate Test-oracle best checkpoints to match this diagnostic
series. This selection protocol is not suitable for claiming formal gains.
No new hyperparameter search or persistent50/50 training is authorized.

## Execution

Server biggpu; host GPU0 UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 only.
Remote root /data1/yb/remote_experiments/osram_cfg84_history_input_gate_20260929.
Separate immutable code snapshot, one-epoch smoke and full runs directories.
Original configs /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full.
Python /data2/yb/reproduction_workspace/envs/s0/bin/python.
Initial GPU free memory31755MiB; /data1 free36GB; previous comparable seed
checkpoint directory451MB. Max3 simultaneous tasks, unchanged batch size.

From the isolated code root, run:

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_cfg84_history_input_gate_20260929/run.py --preflight
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_cfg84_history_input_gate_20260929/run.py --seed 66 --smoke --output-root /data1/yb/remote_experiments/osram_cfg84_history_input_gate_20260929/smoke
/data2/yb/reproduction_workspace/envs/s0/bin/python experiments/osram_cfg84_history_input_gate_20260929/run.py --launch --max-tasks-per-gpu 3
```

For direct --seed smoke set CUDA_VISIBLE_DEVICES=0 and GCNET_DATASET_ROOT=
/data2/yb/paper/GCNet_repro_cmumosi_10seed_20260819/dataset beforehand.
Launcher sets these for its children. It refuses existing result directories.
Coordinator status, child PIDs, effective configs, hashes and alpha diagnostics
are saved under the run root. Completion requires successful child exits and
all eight best checkpoints plus canonical evaluation-mask verification.

## Acceptance before full launch

Verify default-off behavior, original shared initialization/RNG, initial
alpha=1/output identity, scalar bounded scaling only of history, inactive Gap
and padding safety, finite gradients and joint parameter updates. Run a real
MOSI one-epoch smoke on GPU0 before full training. Record actual verification
and launch status separately; passing tests does not establish performance.
