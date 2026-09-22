# Joint-pretrained projector initialization control

This is the corrected control for the three-dataset joint-pretraining
diagnostic. `P_A/P_T/P_V` are loaded from the utterance-level joint JEPA
checkpoint, but remain trainable during MOSI emotion training. There is no
missing-latent reinjection in this control.

The causal OSRAM cfg84 family, cyclic masks, optimizer, 100-epoch budget, and
per-seed/per-rate Test-oracle selection are inherited from the matched MOSI
control. This is an internal diagnostic, not a formal paper result.

Run on biggpu with:

```bash
/data2/yb/reproduction_envs/s0/bin/python \
  experiments/osram_joint_pretrained_init_control_20260922/run.py --launch
```
