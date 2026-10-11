# Approved round1/round2 subset cleanup

User approved the previously listed12round1 and11round2 candidates. This does
not authorize deletion of the entire two batches. Exact23method allowlist in
cleanup.py matches ../checkpoint_cleanup_priority11_20261011/NEXT_CANDIDATES.md.

Preflight verified23complete runs, outputs_verified=true,839weight files totaling
49,937,752,235bytes (46.508 GiB). Before unlinking, recheck completed status,
config/metrics and8predictions, reject symlinks and live Python references;
hash every checkpoint and retained run evidence file into a durable manifest.
After deletion verify every target absent and retained evidence unchanged.

Preserve configurations, histories, logs, predictions, metrics, source snapshots,
reports and all other models. No backups of deleted weights; hashes cannot
restore them. Later inference/resume requires external copies or retraining.
No new training/inference authorized or launched by this cleanup task.
