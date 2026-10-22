# MOSI capacity screen: six runs launched

INTERNAL TEST-ORACLE SCREEN

Seed66 only,150epochs,LR3e-4(1..100)/3e-5(101..150),original GCNet random missing,
official MOSI features/split,s0 environment. All runs start fresh and train jointly.
CHM,LocalSkip,Flat output1600 and task/loss unchanged.

| Model | Flat hidden | V3 width / PairMLP | Total params | Formal result |
|---|---:|---|---:|---|
| f512_plain | 512 | off | 7,032,929 | pending |
| f512_v128 | 512 | 128 / 256 | 7,864,639 | pending |
| f512_v256 | 512 | 256 / 512 | 9,349,567 | pending |
| f1024_plain | 1024 | off | 10,080,865 | pending |
| f1024_v128 | 1024 | 128 / 256 | 10,912,575 | pending |
| f1024_v256 | 1024 | 256 / 512 | 12,397,503 | pending |

Six real-MOSI two-epoch/four-update smokes passed before formal launch. They
verified same-width Flat zero-init equality,finite/nonzero task gradients,
single CHM scan and complete optimizer/RNG/prediction restoration. Smoke scores
are NOT performance evidence. See VALIDATION.md and LAUNCH.json.

Source commit:`86e3900d7fb5f3ee03ce214362e0fececb42887e`.
Three formal jobs on biggpu GPU3,three onGPU7. Controller automatically writes
completed-only RESULT.md/REPORT.json/PER_RATE.csv under:

```text
/data2/yb/remote_experiments/osram_mosi_v3_capacity_20261022/execution_v1
```

Primary comparison: each V3 against its own same-width Flat. Read-only historical
large Flat66 has79.2491 eight-rate WF1/73.4098 high; default V36679.1394/73.7128.
Do not substitute old conversation-mask-v2 scores near80.7.

At the same rate,six models evaluate identical saved per-utterance availability.
Different rates have different masks;training masks are not frozen. Each epoch
must match original training-mask/order/update/LR traces. Test-best per rate may
select different epochs; ACC/MSE come from corresponding same-call predictions.
This is not a single-model deployment score or independent generalization claim.

All new/old checkpoints,smoke weights,masks,predictions and logs remain. No
additional seeds,HPO or mechanism claims. Formal gains/losses are pending.
