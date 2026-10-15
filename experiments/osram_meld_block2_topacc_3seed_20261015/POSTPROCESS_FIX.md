# Exact Test-best recovery after cross-seed auxiliary replay failure

INTERNAL TEST-ORACLE THREE-SEED REPLICATION.

The original six jobs retain immutable training source `ab3d7e6`. Do not restart
their training, alter missing masks, change weights, or erase the failed worker
summaries. Those summaries report a post-training failure, not loss of training.

## Root cause

`gcnet_modality_jepa/train_gcnet.py::_build_protocol_loader` uses an
`EpochSeededSubsetSampler` for all splits, including the official MELD test set.
Its order seed derives from the master seed. Thus seed67/68 enumerate the same
official test utterances in a different conversation order than seed66.

The seed66-only screen's `worker.paired_replay` scatters its canonical saved mask
by a positional cursor, then asserts labels against that same seed66 order.
Extending the worker's seed admission guard did not make this positional replay
cross-seed safe. This oversight is in the replication integration. The label
assertion correctly rejected the invalid auxiliary replay rather than producing
plausible but mismatched paired scores. It does not invalidate Test-best metrics
saved during each seed's own evaluation loop.

## Bounded repair

The user requested three-seed **Test-best ACC**, not another common-mask replay.
Recover the requested metrics on CPU from each existing BEST checkpoint's exact
selected-epoch artifact. Check artifact SHA256, its ACC against the checkpoint,
the full utterance key set, labels and speaker per `(conversation_id, t)`, legal
GCNet endpoint masks, and all50 completed training epochs. Compare identity by
key rather than by position. No training, new inference, label changes, mask
sampling or threshold selection occurs.

Write recovered reports to `testbest_recovery_v1/`, separate from the original
queue/failed summaries. Explicitly mark the auxiliary paired replay as not
completed. Preserve its error and original summary path in recovered provenance.
The future replication entry uses `--testbest-only` so it does not accidentally
invoke the seed66-only positional diagnostic. Screen/HPO defaults remain full
paired replay, and default seed66 behavior is unchanged.

Three regression cases cover a valid shuffled identity order, labels attached to
the wrong IDs, and duplicate IDs. Existing replication contracts also require
the explicit Test-best-only flag. The initial missing implementation/flag failed
the tests; all8 relevant tests pass after the change.

Run the CPU-only recovery (waits only for the remaining authorized jobs):

```bash
CUDA_VISIBLE_DEVICES='' /data2/yb/conda_envs/cmerc-repro/bin/python -u \
  -m experiments.osram_meld_block2_auto_20261014.recover_testbest \
  --root /home/qinjinghui/remote_experiments/osram_meld_block2_topacc_3seed_20261015 \
  --wait
```

Final recovered results must be read from `testbest_recovery_v1/RESULT.md`, not
from the original failed auxiliary replay status or its incomplete summary.
