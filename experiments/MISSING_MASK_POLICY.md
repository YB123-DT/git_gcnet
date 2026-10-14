# Missing-mask protocol policy

For all new headline MELD and IEMOCAP comparisons, use the original GCNet
batch-level sampler:

```text
missing_mask_protocol = gcnet-random-mask-v1
```

The other registered protocols are retained only for reproducing already
completed experiments:

- `conversation-mask-v2`: earlier project-specific screens.
- `gcnet-legacy-fixed-v1`: deterministic replay of retained historical MELD
  checkpoints and frozen interventions. It must not be used for new training
  comparisons.

Every new runner must assert the canonical protocol before starting. Results
from different protocols must not be placed in the same comparison table.
