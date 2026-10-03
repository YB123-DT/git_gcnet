# Same-trajectory base/full task supervision

Implementation and correctness validation only; no new formal training run or
performance result is authorized/claimed in this change.

## Fixed contract

`base` is the original **Local + Memory Flat**, not Local-only. Encoder, causal
Memory, Base/Gap reads, Local Skip and Flat adapter run once on one missing mask.
In particular, the adapter's training dropout is sampled once and its same output
is reused in both readouts:

```text
u = local_skip(Local) + emotion_adapter([Local, Base, masked Gap-A/T/V])
base_hidden = emotion_norm(u)
full_hidden = emotion_norm(u + relation_residual)
base_logits = original_task_head(base_hidden)
full_logits = original_task_head(full_hidden)
loss = 0.5 * original_task_loss(base_logits, labels)
     + 0.5 * original_task_loss(full_logits, labels)
```

Both heads refer to the same module and parameters, as do both emotion_norm calls.
No detached anchor, frozen path, second missing view, second Memory trajectory,
JEPA, completion, contrastive loss, relation classification, or threshold routing.
Task masks, label handling and reduction remain those of the original task loss.
Base supervision reaches the shared upstream path; only full supervision reaches
the Relation branch. Both losses contribute to one backward/optimizer step.

## Switch and inference

Optional `--osram-relation-dual-readout`, default off, used together with
`--osram-relation-block --osram-readout-fusion flat` under cfg84 causal no-JEPA.
The first supported configuration uses original shared task head and sample-mean
task loss, with original cyclic single-view missing masks. The coefficients are
fixed at0.5/0.5; there is no weight search parameter.

Evaluation remains **full only**, preserving the existing W-F1 checkpoint-selection
path. No extra base prediction is needed in normal inference. The training change
adds no parameters and does not modify existing checkpoint tensors or state keys.
Old single-readout behavior remains when the new switch is off.

## Interpretation and scope

This objective explicitly supervises the original Memory-containing path while
training the residual jointly. It is a hypothesis for reducing the original-path
degradation seen in the residual-off audit, not a guarantee of improved W-F1.
It does not freeze the old Flat checkpoint or mathematically constrain its outputs
to remain unchanged during training.

No seed66 or multi-seed performance run is started by this implementation task.
Future experiments must use independent output directories and preserve the original
baseline optimizer/data/mask/checkpoint protocol. Test-oracle outputs, if used for
continuity with prior screening, must remain INTERNAL DIAGNOSTIC ONLY.

## Verified implementation

- `gcnet_missing_m3/osram.py`: reuse the same computed `flat_anchor`; training-only
  base hidden uses the existing emotion_norm. Temporary base output is reset on
  every forward; this is not a temporal history cache.
- `gcnet_missing_m3/model.py`: both outputs use the same `smax_fc`; return API stays
  unchanged. Base logits are exposed only for training and padding is safely masked.
- `gcnet_missing_m3/train_gcnet.py`: config/CLI, fixed dual task loss and detached
  base/full/combined loss logs; original full-output W-F1 selection is unchanged.
- Evaluation model builder propagates the default-off flag. No new state_dict
  entries or parameter initialization are introduced.
- Both global and emotion readout ablations must be `full` in this first version,
  preventing accidental Local-only base supervision.

Target verification on biggpu GPU5/V100, isolated snapshot
`/data2/yb/remote_experiments/osram_relation_dual_readout_check_20261003/code`:

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 \
/data2/yb/reproduction_workspace/envs/s0/bin/python -m unittest \
tests.test_relation_dual_readout tests.test_relation_dual_training \
tests.test_current_history_relation tests.test_memory_shift_integration -v
```

18 tests passed, none skipped, including actual cfg84 CUDA dimensions. CPU legacy,
epoch-integration and paired-view regression checks additionally passed7 tests.
These are synthetic correctness/optimizer checks, not dataset training experiments.

Measured calls per training forward: Encoder1, causal scan1, Local Skip1,
Flat adapter1, shared emotion_norm2, shared task head2. Actual train_epoch over2
batches produces2 model forwards,2 scans and2 optimizer steps; original masks match.
Logged loss equals0.5base+0.5full, with zero JEPA loss and no paired-view metrics.

At zero residual initialization, base/full valid predictions agree; the shared
path gradient agrees with the original loss gradient, while the Relation gradient
is half the former full-only Relation gradient (by design). Base-only backward
reaches Memory query parameters and gives no Relation parameter gradients. Removing
Memory inputs changes base predictions in the test. Joint updates remain finite.
Train/eval full predictions and RNG are unchanged by the new flag at initialization;
the old flag-off path still matches the pre-Relation source regression test.

No performance improvement is claimed. The formal experiment status remains
**not started**.
