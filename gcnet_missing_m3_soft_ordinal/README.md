# Missing-M3 Soft-Ordinal

This folder is the isolated CMU-MOSI soft-ordinal task-head version.

It owns only the version entry point and version-specific tests. The model,
dataset, missing-mask protocol, Slot encoder, JEPA objective, GCNet backbone,
optimizer, and evaluation lifecycle remain shared with `gcnet_missing_m3`.

Entrypoints:

```text
python -m gcnet_missing_m3.train_gcnet
    Existing version; defaults to regression.

python -m gcnet_missing_m3_soft_ordinal.train_gcnet
    Treatment version; locks the MOSI task mode to soft-ordinal.
```

The treatment predicts one signed logit, trains it with an ordered soft binary
target derived from the continuous MOSI label, and always classifies with the
fixed zero-logit threshold. It does not copy or replace the shared backbone.
