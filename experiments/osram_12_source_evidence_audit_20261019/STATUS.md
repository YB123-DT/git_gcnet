# Execution status, 2026-10-19T03:57:20Z

INTERNAL DIAGNOSTIC ONLY

Implementation source: `cb02830e794297e9d3654b5f0c9a4900c6ac0cdf`, pushed to
YB123-DT/git_gcnet, branch feature/ecerc-features. The same source must be used
for both finite jobs. No historical experiment, weight, prediction or mask was
overwritten. Unrelated `.omx/tmux-hook.json` was left untracked and unchanged.

| Stage | Actual status |
|---|---|
| Existing OSRAM/tracker/accepted registries reviewed | Completed |
| Audit-only Base3 capture and strict-past state norms | Implemented |
| Twelve final-input removals and eight source-group coalitions | Implemented |
| Active-row metrics, cancellation, Shapley, requested strata | Implemented |
| Immutable finite two-job launcher and complete-run reports | Implemented |
| Local tests and bounded independent review | Completed,74 tests passed |
| biggpu connection/resource preflight | BLOCKED: SSH banner handshake timeout |
| Remote immutable snapshot deployment | NOT performed |
| Real MELD/IEMOCAP GPU audit | NOT started |
| Real-data4x3 tables and task conclusions | NOT available |

Several separate BatchMode SSH attempts used10--30second connect timeouts,
keepalives, and one alternative IPQoS setting. They failed before executing the
remote preflight (last error: `Connection timed out during banner exchange`).
Do not interpret this as GPU OOM or a model failure. No remote GPU process or
waiting daemon was launched; connection recovery alone will not auto-start a job.
No other user's process was inspected/killed/paused/modified during failed logins.

## Protected planned anchors

These values were resolved from existing accepted registries, not freshly
verified against unreachable remote files. The runner verifies actual SHA256,
strict checkpoint config, dataset and selected-mask identity before evaluation.

- MELD seed66/fold1, original lr1e-5/wd1e-5 Flat:
  `/home/qinjinghui/remote_experiments/osram_meld_wd_seed66_20261018/runs_v1/S66_WD1/best_miss_0p3.pt`
  SHA256 `6f031b515723b441129e16a2e6217410e0eceee2c468e1cbe7b7c3d8276bdb7d`.
- IEMOCAPSix seed66/fold5 original Flat:
  `/home/qinjinghui/remote_experiments/osram_iemocap_flat_seed66_restore_20261016/runs/iemocap6/seed_66/fold_5/best_miss_0p3.pt`
  SHA256 `6a7968ade42513ee868e6dc82f306274a2e7bd53d1943c05f00c1a75af530fd0`.

Both anchors remain fixed for random0.0--0.7 and Persistent A/T/V/AT/AV/TV/ATV.
No new checkpoint selection, training, Adapter or input-mask sampler is added.

## Next execution step, once SSH works

Deploy a git-archive of source cb02830 to an independent remote source directory,
check free space/GPU health, run the documented launcher inside tmux with:

```bash
/data2/yb/conda_envs/cmerc-repro/bin/python -u -m experiments.osram_12_source_evidence_audit_20261019.launch \
  --source-commit cb02830e794297e9d3654b5f0c9a4900c6ac0cdf \
  --gpu-uuid GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049 \
  --output /home/qinjinghui/remote_experiments/osram_12_source_evidence_audit_20261019/runs_v1
```

The UUID is a previously recorded physicalGPU1 mapping; recheck it before launch.
The launcher rejects physicalGPU4 or insufficient memory. Both workers verify
the required cmerc-repro environment and original anchor parity. Recovered outputs
must be complete with matching identity/artifact hashes before analysis/ranking.

Do not rerun training to work around a network failure or report synthetic unit
test errors as the real-data source reconstruction errors. Storage/reading/
decision conclusions await the requested actual frozen inference.
