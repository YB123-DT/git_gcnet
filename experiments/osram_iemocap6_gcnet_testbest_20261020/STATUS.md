# Status: implementation and targeted cleanup audit in progress

Approved finite rerun:30fresh IEMOCAP-6 runs,100epochs,GCNet missing,
per-rate TEST ACC BEST, same-call UA/W-F1. Original full training folds restored.
No formal IEMOCAP rerun is launched until the worker correctness and realGPU
smoke pass. Existing MOSI reruns remain independent and untouched.

Cleanup audit is read-only so far. Only exact erroneous validation-selected
IEMOCAP model files will be deleted; configs/history/predictions/masks and
protected originalTest-best baseline remain. No completed performance claimed.
