# Coalition Evidence Decomposition (CED)

INTERNAL DIAGNOSTIC ONLY — running seed66 per-rate BEST Test-oracle screening.
No performance improvement claim before completion.

Started 2026-10-05 09:37 UTC from immutable code f6e6de3 on biggpu GPU6.
PID 2808863, tmux `osram_ced_seed66_20261005`. Real optimizer step verified
(epoch1/batch1/steps1), peak allocated 1163.58 MiB, reserved1284 MiB.
Per-rate scores remain pending. Launch metadata is in LAUNCH.json.

## Verified implementation

10 focused CPU tests passed on biggpu (9.02s), including two real-model
training steps, finite gradients, shared dropout, exact reconstruction,
inactive/padding safety and default-off identity. Read-only specification and
quality reviews passed. No separate GPU smoke or new dependency.

Measured actual cfg84 parameters: baseline 13,509,793; CED 13,907,879;
added 398,086. Diagnostics stored in epoch history describe the last training
batch, not an epoch-wide aggregate. Performance results remain pending.

## User-defined mechanism

For each current utterance, evaluate F(S) for every nonempty subset S of its
actually observed set O. All subsets use the same current raw observations,
shared original projectors and shared original fusion network. F(empty)=0 by
definition, not by invoking an encoder on an empty mask.

I(S)=sum_{T subset S}(-1)^(|S|-|T|)F(T), retaining the exact first-, second-
and third-order vectors. Separate slots:

```
c1 = active-mean P1(I(A)), P1(I(T)), P1(I(V))
c2 = active-mean P2(I(AT)), P2(I(AV)), P2(I(TV))
c3 = P3(I(ATV)) iff A/T/V all observed, else zero
node_CED = Pout(concat(c1,c2,c3,availability))
```

P1/2/3 are separate LayerNorm256 -> Linear256,256 -> GELU.
Pout is LayerNorm771 -> Linear771,256 -> GELU. Normal initialization; no
zero-initialized residual, no shortcut sum back to F(O). Padding output is zero.
Missing-order slots are zero after all bias-containing transforms.

Current only AV: F(A), F(V), F(AV); no Text evaluation or third-order slot.
Current only A: only F(A). Full ATV: seven nonempty coalitions. Shared modality
projection is computed once per actually observed modality, reused across subsets.
Fusion dropout uses the same per-utterance realization across all subsets;
otherwise stochastic differences could masquerade as interactions.

## Integration and scope

Optional `--osram-ced-block`, default off. Baseline integration parent f6a6f5f.
The CED node replaces the ObservedSetEncoder output entering causal OSRAM.
Original real modality latents remain supplied to the Value path. Node-dependent
keys/queries/local-path inputs change as a consequence: do not claim their
realized tensors or Memory trajectory stay identical. Original Memory read/write
equations, causal scan, Flat readout and task head remain unchanged.

Original optimizer/lr/batch/random cyclic masks/.0–.7 evaluation/100 epochs
are copied from exact Flat seed66 config. Task remains original MSE only.
No extra loss, Gate, attention, completion, JEPA, persistent mix or second view.
One original Memory scan; up to seven current-utterance fusion evaluations,
not seven separate history trajectories. Extra computation is recorded honestly.

Higher-order vectors describe the chosen encoder set function. Mean aggregation,
pattern embeddings, nonlinearity and biases can all contribute to them; they
are not automatically task-relevant synergy, conflict or sentiment effects.
Exact algebra/reconstruction is correctness evidence, not a performance proof.

## Run

Server biggpu, GPU6 UUID GPU-e4cafb17-818e-216a-b94a-7440063a9153 only.
Independent outputs `/data2/yb/remote_experiments/osram_ced_20261005/seed_66`.
Reuse original Flat completed checkpoint scores, do not rerun Flat:
8-rate W-F1 81.068095%, high (.5/.6/.7)76.352251%.

```
python -m experiments.osram_ced_20261005.run --seed 66
  --reference /data2/yb/remote_experiments/osram_mosi_memory_gap_ablation_20260920/full/seed_66/config.json
  --data-manifest /data2/yb/remote_experiments/osram_meaningful20_round2_20261004/DATA.json
  --output /data2/yb/remote_experiments/osram_ced_20261005/seed_66
  --gpu-uuid GPU-e4cafb17-818e-216a-b94a-7440063a9153
```

Bind CUDA_VISIBLE_DEVICES to that exact UUID, use the existing remote s0 Python
and original GCNET_DATASET_ROOT. Immutable source and PID in LAUNCH.json.
Runner saves config/data/source provenance, history, diagnostics, last_training.pt
and eight BEST checkpoints/predictions. No automatic sweep/multi-seed expansion.
