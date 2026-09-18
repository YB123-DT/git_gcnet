# MOSI binary-task objective diagnostic

This is an internal diagnostic, not a formal paper result. It tests whether the
MOSI robustness gap is caused by optimizing continuous regression while
reporting weighted F1 after thresholding.

## Controlled change

The binary run changes only `mosi_task_mode`:

```text
regression -> binary
```

The inherited no-JEPA causal OSRAM configuration is otherwise preserved:

- causal forward-only OSRAM;
- `osram_write_step=0.6`;
- Flat readout and cyclic missing-rate schedule;
- frozen input features and the same test masks;
- same optimizer, learning-rate schedule, batch size, epoch budget, and seeds;
- no JEPA, predictor, completion feedback, or architecture changes.

The binary trainer uses cross-entropy on `1[y > 0]`; original MOSI neutral
labels (`y == 0`) remain excluded according to the existing binary protocol.
For pattern audits, the preserved `continuous_labels` field is used to exclude
original neutral samples, while binary `labels` (0/1) are used for F1.

## Selection and execution

Each seed and missing rate independently selects its highest Test W-F1 epoch.
This is the existing internal Test-oracle diagnostic protocol; it is not a
validation-selected paper result. Seeds 66--70 completed successfully on GPUs
5, 5, 6, 6, and 7.

The accompanying `RESULT.md`, CSV files, and `summary.json` were regenerated
after correcting the binary pattern-audit label handling.

