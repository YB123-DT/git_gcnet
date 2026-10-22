# MOSI Flat512 + V3-128 direct-read variant

INTERNAL TEST-ORACLE SCREEN. One seed66 run only, not a new method search.

The original CHM executes once. V3 takes the12 Value-source/read-role terms and
decodes four forward512 vectors. On valid utterances with previous observations
and at least one currently missing modality, these decoded vectors REPLACE Base
and active Gap inputs to Flat; original reads are NOT added back. Local and
Local Skip are original. Flat hidden512/output1600 and task head remain. V3's
internal projected/message and attention residuals are NOT removed. ATV, no
history, inactive Gap, padding and read-ablation guards retain their old behavior.

Use `osram_source_role_v3=True, osram_source_role_v3_mode='direct'`, V3 width128,
PairMLP256/four attention heads. Default mode `residual` is unchanged. Direct
decoders use ordinary random Linear initialization; residual decoders stay zero.
Neither mode adds parameters: total7,864,639; V3831,710; Flat adapter3,058,240.
This compares direct/random-init with residual/zero-init, not add-back alone.

100epochs,constant3e-4,batch32,Adam,WD1e-5,MSE,cyclic0..0.7,
gcnet-random-mask-v1,original splits/features and archived s0 environment. Exact
Flat66 test masks reused, original per-epoch training mask/order/step/LR traces
checked. Existing150-epoch controls are reselected using first100 logs only,
without new inference or training. Per-rate Test-best W-F1 selects8BEST with
matched ACC/MSE, predictions/masks and full last model/Adam/RNG. Smoke artifacts
are preserved. No additional seed, HPO, Gate, JEPA, completion or auxiliary loss.

```bash
/data2/yb/reproduction_workspace/envs/s0/bin/python -u \
  -m experiments.osram_mosi_v3_direct_20261022.run \
  --source-commit SOURCE_SHA \
  --gpu-uuid GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e \
  --output /data2/yb/remote_experiments/osram_mosi_v3_direct_20261022/execution_v2
```

The tmux controller runs one two-epoch/four-update smoke, requires finite/nonzero
V3 gradients and model/Adam/RNG/prediction restoration, then starts one fresh
100-epoch run only if measured GPU/RAM/disk headroom allows. It never signals
other jobs or changes batch/precision/config. Worker supports explicit same-source
`--resume`; controller never silently retries or overwrites an output directory.

Initial task predictions can still equal Flat because its adapter starts at
zero; this does NOT imply identity of V3's randomly replaced history inputs.
Actual Flat input replacement is verified in core tests, not inferred from logits.

No independent generalization or guaranteed performance improvement is claimed.

Actual immutable training source:9827f30df1c48c758ae714ae35e044cdd0caea5f,
directory `/data2/yb/remote_experiments/osram_mosi_v3_direct_20261022/source_retry`.
tmux socket/session:`osram_mosi_direct` / `direct_9827f30`. PhysicalGPU7 is bound
by the UUID in the command; process-local device is cuda:0. ControllerPID3346065,
trainingPID3349620 were verified alive, with the training process on the correct
UUID. `LAUNCH.json` records the timestamp/config/environment and actual GPU proof.
execution_v1/source_b93209e preserve the pre-update metadata-check failure; they
are not formal results and are never mixed into the ranking.
