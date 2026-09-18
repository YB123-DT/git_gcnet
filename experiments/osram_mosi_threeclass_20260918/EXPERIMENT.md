# MOSI three-class-training / Non0 binary-test diagnostic

This is an internal diagnostic, not a formal paper result. It tests whether
including MOSI neutral utterances as a learned class can improve the final
negative-versus-positive metric under missing modalities.

## Training and evaluation labels

Training uses three-class cross entropy:

```text
y < 0  -> class 0 (negative)
y == 0 -> class 1 (neutral)
y > 0  -> class 2 (positive)
```

At test time, original continuous `y == 0` samples are excluded. For every
remaining sample, only the class-0 and class-2 logits are compared, producing
the reported Non0 Accuracy and weighted F1. The neutral logit is therefore
used during training but not as a reported test class.

## Controlled protocol

The inherited no-JEPA causal OSRAM is unchanged: forward-only OSRAM,
`osram_write_step=0.6`, Flat readout, cyclic missing-rate schedule, frozen
features, optimizer, masks, and per-seed × per-rate Test-oracle checkpoint
selection. Only `mosi_task_mode` changes from `regression` to `three-class`.

Seeds 66--70 ran concurrently on GPUs 5, 5, 6, 6, and 7. The results remain
diagnostic because checkpoint selection uses the test split.

