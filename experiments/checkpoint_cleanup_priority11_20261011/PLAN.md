# Partial checkpoint cleanup of historical module screening

User authorized deleting part of the older hundred-module checkpoint collection,
retaining evidence. Narrow scope: eleven completed seed66 priority40 candidates
with mean8 W-F1 below80.3%. This is a storage decision, not removal of negative
results from reporting. Preserve all metrics, predictions, configs, logs, source,
reports and original provenance. Do not touch all other candidates or seeds.

|ID|Method|Mean8 W-F1 (%)|
|---|---|---:|
|M01|SAB|80.035|
|M02|DAT|80.203|
|M05|PNA|80.283|
|M07|LMF|80.271|
|M18|FSPool|80.175|
|M21|NFM|79.610|
|M22|DCNv2|80.081|
|M24|AFM|80.267|
|M30|DyT|80.108|
|M35|DynaMixer|79.908|
|M37|entmax-SAB|80.101|

Remote exact run paths are hardcoded in cleanup.py and expanded in MANIFEST.json.
Preflight: complete/outputs_verified,8predictions,config/metrics, no matching live
Python job; forbid symlinks. Hash each target and every retained run evidence
file before deletion. Delete only379explicit weight files (20.478 GiB), no
recursive directory removal. Recheck evidence hashes and target absence.
Manifest is an audit trail, not a weight backup. Deleted weights require another
copy or retraining for subsequent inference/continuation. Other high-scoring
candidates, all baselines and cross-dataset/multi-seed/150epoch runs are excluded.

INTERNAL DIAGNOSTIC ONLY applies to the performance figures; no new training,
inference or claim of statistical inferiority is made by this storage cleanup.
