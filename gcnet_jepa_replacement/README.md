# Clean GCNet JEPA Replacement

This package defines the version in which modality JEPA replaces the original
GCNet reconstruction head. It does not instantiate `linear_rec`, returns no
reconstruction tensors, and cannot be combined with `--loss-recon`.

The training objective is:

```text
classification cross entropy + jepa_weight * centered modality prediction
```

Run through the shared trainer with:

```bash
python -m gcnet_modality_jepa.train_gcnet \
  --model-variant replacement \
  --jepa-weight 0.1 \
  ...
```

Do not pass `--loss-recon` for this variant.
